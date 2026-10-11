# 阶段三·单卡 CUDA、AMP 与 activation checkpointing

状态：**已实现并可复现**。这是当前唯一保留的阶段三工程：单卡 CUDA 使用 `--amp --activation-checkpointing`，语料只包含维基摘要、斯瓦希里语和 Python 代码，明确排除英法歌曲。它拥有独立的原始语料、tokenizer、输出目录和 checkpoint。

## 工程内容

- `config.json`：0.1B profile、AMP、activation checkpointing、梯度累积和 seed。
- `scripts/train.py`：单卡预训练循环，支持 AMP、梯度累积、激活重计算和分片最佳 checkpoint。
- `scripts/data_pipeline.py`：仅扫描本目录 `data/raw/`，按文件名排序并写入 corpus fingerprint。
- `data/raw/`：`wiki_language_model.txt`、`wiki_transformer.txt`、`swahili_language.txt`、`code_python_subset.py`；没有英法歌曲文件。
- `logs/`：配置、训练 CSV、独立 PPL 和实验记录。
- `curves/training_loss.png`：真实训练曲线。
- `models/best_checkpoint.json` + 9 个分片：验证 PPL 最佳模型。

## 一键复现

从仓库根目录执行：

```powershell
. ..\Use-ProjectEnv.ps1
..\.venv\Scripts\python.exe 03_pretraining/cuda_amp_checkpointing/scripts/train.py --profile 0.1b --steps 50 --eval-interval 10 --eval-iters 2 --device cuda --amp --activation-checkpointing --block-size 128 --batch-size 1 --grad-accum 4
..\.venv\Scripts\python.exe 03_pretraining/cuda_amp_checkpointing/scripts/evaluate.py --checkpoint 03_pretraining/cuda_amp_checkpointing/models/best_checkpoint.json --corpus 03_pretraining/cuda_amp_checkpointing/data/raw --iters 4 --batch-size 1 --device cuda --output 03_pretraining/cuda_amp_checkpointing/logs/evaluation_ppl.json
..\.venv\Scripts\python.exe 03_pretraining/cuda_amp_checkpointing/scripts/check_repro.py
```

单卡 CUDA的 `evaluate.py` 默认只支持 PPL；它拒绝没有本目录语料依据的英法翻译任务，确保阶段三工程不发生数据混用。

## 实际结果

模型为 12 层、12 heads、768 hidden，参数量 85,547,520；共 8,851 tokens，50 个 optimizer steps，`amp_enabled=true`，`activation_checkpointing=true`。验证 PPL 从 594.85（step 0）降至 134.15（step 50）；独立评测 PPL 为 121.31。

详见 [`logs/experiment.md`](logs/experiment.md)、[`logs/training_metrics.csv`](logs/training_metrics.csv) 和 [`curves/training_loss.png`](curves/training_loss.png)。
