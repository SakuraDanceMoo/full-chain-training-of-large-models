"""Correctness-first CUDA attention benchmarks and real torch.profiler capture."""
import argparse
import csv
import html
import json
import platform
import statistics
import sys
import warnings
from pathlib import Path

import torch
import triton
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from attention_kernel import attention

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE / 'third_party'))
from flash_attention_reference import attention as reference_attention


def flash_reference(q, k, v):
    if q.shape[-2] < 128 or q.shape[-2] % 128:
        raise RuntimeError('Official v3.2 tutorial requires N divisible by 128 for safe full-tile access')
    return reference_attention(q, k, v, True, q.shape[-1] ** -0.5)


def eager(q, k, v, causal=True):
    scores = q @ k.transpose(-2, -1) * (q.shape[-1] ** -0.5)
    if causal:
        mask = torch.ones(q.shape[-2], q.shape[-2], device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, -float('inf'))
    return scores.softmax(-1) @ v


def sdpa(q, k, v, backend, causal=True):
    # Force a backend: never silently label a math fallback as FlashAttention.
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=0.)


def correctness():
    torch.manual_seed(20261019)
    rows = []
    for dtype in (torch.float16, torch.bfloat16):
        for d in (32, 64, 128):
            for n in (1, 17, 63, 64, 65, 127, 129, 257, 511):
                for causal in (False, True):
                    q, k, v = [torch.randn(1, 2, n, d, device='cuda', dtype=dtype) for _ in range(3)]
                    ref = sdpa(q.float(), k.float(), v.float(), SDPBackend.MATH, causal)
                    out = attention(q, k, v, causal).float()
                    torch.testing.assert_close(out, ref, atol=.02, rtol=.02)
                    rows.append(dict(dtype=str(dtype), n=n, d=d, causal=causal,
                                     max_absolute_error=float((out-ref).abs().max())))
    # Future key/value perturbations must not affect earlier query positions.
    q, k, v = [torch.randn(1, 2, 129, 64, device='cuda', dtype=torch.float16) for _ in range(3)]
    before = attention(q, k, v)
    k[:, :, 65:] += 10
    v[:, :, 65:] -= 10
    after = attention(q, k, v)
    torch.testing.assert_close(before[:, :, :65], after[:, :, :65], atol=0, rtol=0)
    return dict(cases=rows, causal_future_invariance=True,
                tolerance=dict(atol=.02, rtol=.02), reference='fp32 SDPA MATH', passed=True)


def timings(fn, warmup=20, samples=25, repeats=10, mode='launch'):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if mode == 'cuda_graph':
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                fn()
        fn = graph.replay
    measurements = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(1 if mode == 'cuda_graph' else repeats):
            fn()
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end) / repeats)
    return measurements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--correctness-only', action='store_true')
    args = parser.parse_args()
    logs = STAGE / 'logs'
    logs.mkdir(exist_ok=True)
    checks = correctness()
    (logs / 'attention_correctness.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print('Correctness PASS:', len(checks['cases']), 'cases + future invariance', flush=True)
    if args.correctness_only:
        return
    environment = dict(torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
                       python=platform.python_version(), os=platform.platform(), gpu=torch.cuda.get_device_name(),
                       gpu_count=torch.cuda.device_count(), timing='CUDA events, warmup=20; 25 samples x 10 calls; CUDA Graph captures 10 calls per replay',
                       scope='forward only; dropout=0; contiguous fp16 B=2 H=12 D=64; causal=True',
                       flash_attention='Official Triton v3.2.0 FlashAttention-v2 tutorial reference; SDPA FLASH also probed explicitly',
                       pytorch_flash_compiled=torch.backends.cuda.is_flash_attention_available())
    (logs / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    rows, unavailable = [], []
    for n in (64, 128, 256, 512, 1024, 2048):
        q, k, v = [torch.randn(2, 12, n, 64, device='cuda', dtype=torch.float16) for _ in range(3)]
        ref = sdpa(q.float(), k.float(), v.float(), SDPBackend.MATH)
        methods = dict(eager=lambda: eager(q, k, v),
                       sdpa_math=lambda: sdpa(q, k, v, SDPBackend.MATH),
                       sdpa_flash=lambda: sdpa(q, k, v, SDPBackend.FLASH_ATTENTION),
                       flashattention2_ref=lambda: flash_reference(q, k, v),
                       triton_32=lambda: attention(q, k, v, block_m=32),
                       triton_64=lambda: attention(q, k, v, block_m=64))
        for name, fn in methods.items():
            try:
                torch.testing.assert_close(fn().float(), ref, atol=.02, rtol=.02)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                allocated = torch.cuda.memory_allocated()
                fn()
                torch.cuda.synchronize()
                extra_bytes = torch.cuda.max_memory_allocated()-allocated
                for mode in ('launch', 'cuda_graph'):
                    values = timings(fn, mode=mode)
                    for sample, ms in enumerate(values):
                        rows.append(dict(method=name, sequence=n, execution=mode, sample=sample, latency_ms=ms,
                                         tokens_per_second=2*n/(ms/1000), peak_extra_bytes=extra_bytes,
                                         effective_tflops=4*2*12*n*n*64*.5/(ms*1e9)))
                    print(n, name, mode, f'{statistics.median(values):.4f} ms', flush=True)
            except RuntimeError as exc:
                unavailable.append(dict(sequence=n, method=name, reason=str(exc)))
                if name.startswith('triton'):
                    raise
    with (logs / 'benchmark_metrics.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (logs / 'backend_unavailable.json').write_text(json.dumps(unavailable, indent=2), encoding='utf-8')

    # JIT compile before capture. Record actual operators and CUDA kernels.
    q, k, v = [torch.randn(2, 12, 512, 64, device='cuda', dtype=torch.float16) for _ in range(3)]
    capture = dict(eager=lambda: eager(q, k, v),
                   sdpa_math=lambda: sdpa(q, k, v, SDPBackend.MATH),
                   sdpa_flash=lambda: sdpa(q, k, v, SDPBackend.FLASH_ATTENTION),
                   flashattention2_ref=lambda: flash_reference(q, k, v),
                   triton=lambda: attention(q, k, v))
    enabled = {}
    for name, fn in capture.items():
        try:
            fn()
            enabled[name] = fn
        except RuntimeError:
            pass
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True, profile_memory=True) as profiler:
        for _ in range(5):
            for name, fn in enabled.items():
                with torch.profiler.record_function(name):
                    fn()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(logs / 'attention_trace.json'))
    table = profiler.key_averages().table(sort_by='self_cuda_time_total', row_limit=25)
    (logs / 'profiler_table.txt').write_text(table, encoding='utf-8')
    events = [dict(name=e.key, self_device_time_us=e.self_device_time_total, count=e.count)
              for e in profiler.key_averages()]
    (logs / 'profiler_events.json').write_text(json.dumps(events, indent=2), encoding='utf-8')
    (logs / 'profiling.html').write_text(
        '<!doctype html><meta charset="utf-8"><style>body{font:16px Consolas,monospace;padding:24px;background:#fff;color:#17324d}pre{font-size:13px}h1{font:24px Arial}</style>'
        '<h1>Stage 4 | Actual torch.profiler capture</h1><p>RTX 4070 Ti | B=2, H=12, N=512, D=64 | fp16 causal forward</p>'
        '<p>5 calls per backend; JIT compiled before capture; raw trace: attention_trace.json</p><pre>' + html.escape(table) + '</pre>', encoding='utf-8')
    print('Profiler trace and HTML saved', flush=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', message='.*attention.*')
    warnings.filterwarnings('ignore', message='Torch was not compiled with flash attention.*')
    main()
