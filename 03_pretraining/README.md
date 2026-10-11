# 0.1B 端到端预训练

状态：**已实现（implemented；单卡 CUDA 0.1B 增量训练已跑通）**。

使用阶段一 BPE、阶段二 GPT，在维基摘要、斯瓦希里语、英法歌曲和 Python 代码子集组成的混合语料上完成端到端数据管线、混合精度/梯度累积/激活重计算、分片最佳 checkpoint、PPL 和英法翻译小任务评测。默认 smoke 配置用于快速验证；`--profile 0.1b` 提供约 0.1B 参数规模配置。

## 目录约定

- `scripts/`：核心实现、训练、评测与阶段验收脚本。
- `data/raw/`：原始英法语料及 `mixed/` 维基/小语种/代码子集；`data/processed/`：token ID 缓存。
- `logs/`：运行输出、配置、指标、PPL/下游评测、架构图和实验记录。
- `curves/`：真实训练 loss 曲线。

## 验收

在仓库根目录执行：

```powershell
python 03_pretraining/scripts/check_stage.py
```

阶段验收脚本检查核心训练、评测、数据管线、配置、loss 曲线和 checkpoint。

预期交付：

- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/data_pipeline.py`
- `logs/training_metrics.csv`
- `logs/ppl_and_downstream.json`
- `logs/evaluation_ppl.json`、`logs/evaluation_translation.json`
- `logs/model_architecture.png`
- `curves/training_loss.png`

## 一键复现

在仓库根目录执行以下命令，默认使用单卡/CPU smoke 配置：

```powershell
python 03_pretraining/scripts/train.py --profile smoke --steps 80 --eval-interval 20
python 03_pretraining/scripts/evaluate.py --checkpoint 03_pretraining/models/checkpoint_step_000080.pt
```

单卡 CUDA 的本次增量复现命令（12 GB 显存采用较短上下文和梯度累积）：

```powershell
python 03_pretraining/scripts/train.py --profile 0.1b --steps 50 --eval-interval 10 --eval-iters 2 --device cuda --amp --activation-checkpointing --block-size 128 --batch-size 1 --grad-accum 4 --corpus 03_pretraining/data/raw/mixed --output 03_pretraining
python 03_pretraining/scripts/evaluate.py --checkpoint 03_pretraining/models/best_checkpoint.json --corpus 03_pretraining/data/raw/mixed --task ppl --device cuda
python 03_pretraining/scripts/evaluate.py --checkpoint 03_pretraining/models/best_checkpoint.json --corpus 03_pretraining/data/raw/mixed --translation-data 03_pretraining/data/raw/eng_fra_songs.txt --task translation --device cuda
```

完整 0.1B 默认 profile 使用上下文 512、batch 2、梯度累积 16；显存不足时先按上面的单卡配置逐步增加 token 数和训练步数。

`--grad-accum` 由 profile 配置控制；AMP 只在 CUDA 上启用，激活重计算通过阶段二 GPT block 的 checkpoint 路径启用。显存不足时优先降低 `--batch-size`、提高梯度累积、开启 `--amp` 和 `--activation-checkpointing`，再降低 `--block-size`。

## 真实产物与限制

本次 CUDA run 使用 39,557 tokens、50 optimizer steps，参数量 85,547,520，AMP 与 activation checkpointing 均开启。最佳验证 PPL 为 147.85（step 30），最佳权重保存为 `models/best_checkpoint.json` 加 4 个小于 100 MiB 的分片。独立评测记录在 `evaluation_ppl.json` 和 `evaluation_translation.json`；后者是 20 个英法句对的 teacher-forced 翻译小任务，属于真实小任务验证，不代表通用翻译 benchmark。完整长程收敛仍需要更大语料和更多步数。

模型架构图：

![Stage 3 model architecture](logs/model_architecture.png)

实验记录见 [`logs/2026-10-11_stage3_pretraining.md`](logs/2026-10-11_stage3_pretraining.md)。

下一阶段：04_systems_scaling。
