"""Forward-only tiled attention with online softmax; no N x N score buffer.

Layout: contiguous [batch, heads, sequence, head_dim], fp16/bf16 CUDA.
This learning kernel implements inference only (no dropout or backward).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _attention(Q, K, V, O, N: tl.constexpr, D: tl.constexpr,
               SCALE: tl.constexpr, CAUSAL: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    rows = query_block * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    dims = tl.arange(0, D)
    base = batch_head * N * D
    q = tl.load(Q + base + rows[:, None] * D + dims[None, :],
                mask=rows[:, None] < N, other=0)
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    normalizer = tl.zeros((BM,), tl.float32)
    numerator = tl.zeros((BM, D), tl.float32)
    upper = N
    if CAUSAL:
        upper = tl.minimum((query_block + 1) * BM, N)
    for start in range(0, upper, BN):
        keys = start + cols
        k = tl.load(K + base + keys[None, :] * D + dims[:, None],
                    mask=keys[None, :] < N, other=0)
        v = tl.load(V + base + keys[:, None] * D + dims[None, :],
                    mask=keys[:, None] < N, other=0)
        scores = tl.dot(q, k) * SCALE
        visible = keys[None, :] < N
        if CAUSAL:
            visible = visible & (keys[None, :] <= rows[:, None])
        scores = tl.where(visible, scores, -float('inf'))
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        correction = tl.exp(maximum - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        numerator = numerator * correction[:, None]
        numerator += tl.dot(probabilities.to(q.dtype), v)
        normalizer = normalizer * correction + tl.sum(probabilities, 1)
        maximum = new_max
    result = numerator / normalizer[:, None]
    tl.store(O + base + rows[:, None] * D + dims[None, :], result,
             mask=rows[:, None] < N)


def attention(q, k, v, causal=True, block_m=64, block_n=64):
    """Compute softmax(QK^T / sqrt(d)) V without autograd support."""
    if q.ndim != 4 or not (q.shape == k.shape == v.shape):
        raise ValueError('Q/K/V must share [B,H,N,D] shape')
    if any(not x.is_cuda or not x.is_contiguous() for x in (q, k, v)):
        raise ValueError('Q/K/V must be contiguous CUDA tensors')
    if any(x.device != q.device or x.dtype != q.dtype for x in (k, v)):
        raise ValueError('Q/K/V must share dtype and device')
    if q.dtype not in (torch.float16, torch.bfloat16) or q.shape[-1] not in (32, 64, 128):
        raise ValueError('Supported: fp16/bf16 with head dimension 32/64/128')
    if min(q.shape) < 1 or any(x.requires_grad for x in (q, k, v)):
        raise ValueError('Nonempty inference inputs required; backward is unsupported')
    if block_m not in (32, 64) or block_n not in (32, 64):
        raise ValueError('Block sizes must be 32 or 64')
    b, h, n, d = q.shape
    out = torch.empty_like(q)
    with torch.cuda.device(q.device):
        _attention[(triton.cdiv(n, block_m), b * h)](
            q, k, v, out, n, d, d ** -0.5, causal, block_m, block_n,
            num_warps=4, num_stages=2)
    return out
