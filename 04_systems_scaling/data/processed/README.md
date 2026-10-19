# 派生数据

- `tokens.npz`：阶段一 BPE 类编码的 int32 token IDs；词表加载自 `03_pretraining/models/tokenizer_bpe_512.json`。
- `split.json`：原文/tokenizer SHA-256、训练/验证位置；验证末尾 8192 tokens，训练最大预算 131072。
- `scaling_cell_summary.csv`：25 个 N×D cell 的 3-seed 均值、样本标准差和预测/留出标签。

由 `run_scaling.py` 和 `fit_scaling.py` 生成。原文或 tokenizer hash 变化时拒绝复用缓存，需明确移除派生缓存再生成；不得覆盖原文。
