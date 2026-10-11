# 阶段三：0.1B 端到端预训练

状态：**已实现并可复现**。当前仅保留阶段三第二部分：单卡 CUDA 训练，启用 `--amp --activation-checkpointing`，使用维基摘要、斯瓦希里语和 Python 代码语料。英法歌曲不在本部分训练数据中。

## 工程内容

目录 [`part2_mixed_amp_checkpointing/`](part2_mixed_amp_checkpointing/README.md) 是完整、独立的预训练工程，包含：

- `config.json`：0.1B profile、AMP、activation checkpointing、梯度累积和 seed。
- `scripts/train.py`：单卡预训练循环，支持 AMP、梯度累积、激活重计算和分片最佳 checkpoint。
- `scripts/data_pipeline.py`：只扫描本目录 `data/raw/`，按文件名排序并写入 corpus fingerprint。
- `data/raw/`：维基摘要、斯瓦希里语和 Python 代码四个文件，不含英法歌曲。
- `logs/`：训练配置、CSV 指标、独立 PPL、终端日志和实验记录。
- `curves/training_loss.png`：真实训练 loss 曲线。
- `models/best_checkpoint.json` + 9 个分片：验证 PPL 最佳模型。

## 一键复现

从仓库根目录执行：

```powershell
. ..\Use-ProjectEnv.ps1
..\.venv\Scripts\python.exe 03_pretraining/part2_mixed_amp_checkpointing/scripts/train.py --profile 0.1b --steps 50 --eval-interval 10 --eval-iters 2 --device cuda --amp --activation-checkpointing --block-size 128 --batch-size 1 --grad-accum 4
..\.venv\Scripts\python.exe 03_pretraining/part2_mixed_amp_checkpointing/scripts/evaluate.py --checkpoint 03_pretraining/part2_mixed_amp_checkpointing/models/best_checkpoint.json --corpus 03_pretraining/part2_mixed_amp_checkpointing/data/raw --iters 4 --batch-size 1 --device cuda --output 03_pretraining/part2_mixed_amp_checkpointing/logs/evaluation_ppl.json
..\.venv\Scripts\python.exe 03_pretraining/part2_mixed_amp_checkpointing/scripts/check_repro.py
```

## 实际结果

模型为 12 层、12 heads、768 hidden，参数量 85,547,520；共 8,851 tokens，50 个 optimizer steps，`amp_enabled=true`，`activation_checkpointing=true`。验证 PPL 从 594.85（step 0）降至 134.15（step 50）；独立评测 PPL 为 121.31。

详见 [`part2_mixed_amp_checkpointing/logs/experiment.md`](part2_mixed_amp_checkpointing/logs/experiment.md)、[`training_metrics.csv`](part2_mixed_amp_checkpointing/logs/training_metrics.csv)、[`training_loss.png`](part2_mixed_amp_checkpointing/curves/training_loss.png) 和 [`best_checkpoint.json`](part2_mixed_amp_checkpointing/models/best_checkpoint.json)。

阶段三根目录下早期聚合脚本和日志保留作历史审计；当前复现实验以 `part2_mixed_amp_checkpointing/` 为准。
