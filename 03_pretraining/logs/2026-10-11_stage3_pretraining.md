# 阶段三端到端预训练实验记录

日期与时区：2026-10-11（Asia/Hong_Kong）  
阶段：`03_pretraining`  
架构图：[`model_architecture.png`](model_architecture.png)

## 做了什么

复用阶段一 `BPETokenizer` 和阶段二 `GPT/GPTBlock`，新增 `data_pipeline.py` 将 `eng_fra_songs.txt` 按英文/法文句对编码为带 `<eos>` 的 token stream，按 90%/10% 划分训练与验证数据。新增 `train.py` 自建预训练循环，记录 loss/PPL/next-token accuracy，定期保存 checkpoint，并支持梯度累积、CUDA AMP 和 activation checkpointing。新增 `evaluate.py` 对 held-out split 计算 PPL 与 next-token accuracy。

`smoke` 配置为 2 层、4 heads、128 hidden、上下文 64、batch 4、梯度累积 2，在 CPU 上先完成闭环；`0.1b` 配置为 12 层、12 heads、768 hidden、上下文 512，作为单卡扩展配置。

## 结果是什么

实际 smoke 运行命令：

```powershell
python 03_pretraining/scripts/train.py --profile smoke --steps 80 --eval-interval 20 --eval-iters 4 --device cpu --no-amp
```

指标保存在 [`training_metrics.csv`](training_metrics.csv)，loss 曲线保存在 [`../curves/training_loss.png`](../curves/training_loss.png)：

| step | train loss | validation loss | validation PPL | validation next-token accuracy |
|---:|---:|---:|---:|---:|
| 0 | 6.2701 | 6.2587 | 522.53 | 0.0020 |
| 40 | 5.2261 | 5.3272 | 205.87 | 0.0596 |
| 80 | 4.8999 | 4.9411 | 139.92 | 0.0820 |

loss/PPL 下降，说明 tokenizer → batch → GPT → 梯度更新 → checkpoint 的闭环可运行。模型 checkpoint 为 `models/checkpoint_step_000080.pt`；配置和实际设备记录在 [`training_config.json`](training_config.json)，轻量评测说明在 [`ppl_and_downstream.json`](ppl_and_downstream.json)。

显存调试路径已实现但本次 CPU smoke 未开启 CUDA AMP：降低 micro-batch、增加梯度累积、开启 `--amp`，再开启 `--activation-checkpointing`。完整 0.1B profile 尚未在本次 CPU smoke 中声称完成收敛。

## 下一步是什么

在单卡 CUDA 上运行 `--profile 0.1b --amp --activation-checkpointing`，逐步增加 token 数和训练步数；接入维基/小语种/代码混合语料，保存最佳验证 PPL checkpoint，并把 held-out next-token proxy 替换为真实下游分类或翻译小任务。
