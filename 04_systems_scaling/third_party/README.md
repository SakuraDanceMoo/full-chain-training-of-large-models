# 官方 FlashAttention-v2 对照来源

`flash_attention_reference.py` 是 Triton **v3.2.0** 官方 `python/tutorials/06-fused-attention.py` 原样归档。2026-10-19 从上游 raw URL 获取，并保留同 tag 的 `LICENSE` 为 `TRITON_LICENSE`。

- https://github.com/triton-lang/triton/blob/v3.2.0/python/tutorials/06-fused-attention.py
- https://github.com/triton-lang/triton/blob/v3.2.0/LICENSE

使用该版本的 block-pointer 内核，可在本机 Ada GPU 和 Triton 3.4.0 上运行，无需较新教程的 descriptor 路径。上游包含测试装饰器，导入需要 pytest。

Benchmark 仅调用 causal FP16 forward，N≥128 且为 128 整数倍；不运行其测试或 benchmark main，不使用其 backward。它不能等同于当前 `flash-attn` 包或 PyTorch 编译内置 FLASH backend。来源及哈希见 [provenance.json](provenance.json)。
