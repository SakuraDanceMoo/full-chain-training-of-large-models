# 阶段三：0.1B 端到端预训练

状态：**已实现并可复现**。本阶段直接在 `03_pretraining/` 下组织单卡 CUDA 预训练工程，启用 `--amp --activation-checkpointing`，训练语料只包含维基摘要、斯瓦希里语和 Python 代码，不含英法歌曲。

## 工程内容

- `config.json`：0.1B profile、AMP、activation checkpointing、梯度累积、seed 与复现命令。
- `scripts/train.py`：单卡预训练循环，支持 AMP、梯度累积、激活重计算和分片最佳 checkpoint。
- `scripts/data_pipeline.py`：按文件名排序读取 `data/raw/`，使用阶段一 BPE tokenizer 并记录语料指纹。
- `scripts/evaluate.py`：独立验证集 PPL 与 next-token proxy 评测。
- `data/raw/`：维基摘要、斯瓦希里语和 Python 代码四个文件；没有英法歌曲文件。
- `logs/`：训练配置、CSV 指标、终端输出、PPL 评测和实验记录。
- `curves/training_loss.png`：真实训练 loss 曲线。
- `models/best_checkpoint.json` 及 9 个分片：验证 PPL 最佳模型。

## 一键复现

从仓库根目录执行：

```powershell
. ..\Use-ProjectEnv.ps1
..\.venv\Scripts\python.exe 03_pretraining/scripts/train.py --profile 0.1b --steps 50 --eval-interval 10 --eval-iters 2 --device cuda --amp --activation-checkpointing --block-size 128 --batch-size 1 --grad-accum 4
..\.venv\Scripts\python.exe 03_pretraining/scripts/evaluate.py --checkpoint 03_pretraining/models/best_checkpoint.json --corpus 03_pretraining/data/raw --iters 4 --batch-size 1 --device cuda --output 03_pretraining/logs/evaluation_ppl.json
..\.venv\Scripts\python.exe 03_pretraining/scripts/check_repro.py
```

单卡 CUDA 的 `evaluate.py` 默认只支持 PPL；本阶段没有英法翻译训练语料，避免不同阶段的数据混用。

## 实际结果

模型为 12 层、12 heads、768 hidden，参数量 85,547,520；共 8,851 tokens，50 个 optimizer steps，`amp_enabled=true`，`activation_checkpointing=true`。验证 PPL 从 594.85 降至 134.15，独立评测 PPL 为 121.31。

详见 [`logs/experiment.md`](logs/experiment.md)、[`logs/training_metrics.csv`](logs/training_metrics.csv)、[`curves/training_loss.png`](curves/training_loss.png)、[`logs/model_architecture.png`](logs/model_architecture.png) 和 [`models/best_checkpoint.json`](models/best_checkpoint.json)。
