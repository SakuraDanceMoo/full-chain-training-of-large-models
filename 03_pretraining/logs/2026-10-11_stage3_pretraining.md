# 阶段三端到端预训练实验记录

日期与时区：2026-10-11（Asia/Hong_Kong）  
阶段：`03_pretraining`  
架构图：[`model_architecture.png`](model_architecture.png)

## 做了什么

复用阶段一 `BPETokenizer` 和阶段二 `GPT/GPTBlock`，新增 `data_pipeline.py` 将维基摘要、斯瓦希里语、英法歌曲和 Python 代码子集按文件排序编码为带 `<eos>` 的 token stream，按 90%/10% 划分训练与验证数据。新增 `train.py` 自建预训练循环，记录 loss/PPL/next-token accuracy，按验证 PPL 保存分片最佳 checkpoint，并支持梯度累积、CUDA AMP 和 activation checkpointing。新增 `evaluate.py` 对 held-out split 计算 PPL，并对英法句对执行 teacher-forced 翻译小任务。

`smoke` 配置为 2 层、4 heads、128 hidden、上下文 64、batch 4、梯度累积 2；本次 CUDA 增量使用 12 层、12 heads、768 hidden、上下文 128、batch 1、梯度累积 4，参数量 85,547,520。

## 结果是什么

当前独立 CUDA 工程的实际运行命令：

```powershell
python 03_pretraining/cuda_amp_checkpointing/scripts/train.py --profile 0.1b --steps 50 --eval-interval 10 --eval-iters 2 --device cuda --amp --activation-checkpointing --block-size 128 --batch-size 1 --grad-accum 4
```

指标保存在 [`../cuda_amp_checkpointing/logs/training_metrics.csv`](../cuda_amp_checkpointing/logs/training_metrics.csv)，唯一的阶段三 loss 曲线保存在 [`../cuda_amp_checkpointing/curves/training_loss.png`](../cuda_amp_checkpointing/curves/training_loss.png)：

| step | train loss | validation loss | validation PPL | validation next-token accuracy |
|---:|---:|---:|---:|---:|
| 0 | 6.3711 | 6.4032 | 603.75 | 0.0000 |
| 30 | 4.6101 | 4.9962 | 147.85 | 0.0117 |
| 50 | 3.6887 | 5.3081 | 201.96 | 0.0195 |

先进行 2-step/20k-token warmup，再增加到完整混合语料和 50 steps。最佳验证 PPL 出现在 step 30，权重保存为 `models/best_checkpoint.json` 和 4 个分片。独立评测的 held-out PPL 约为 208.21；英法翻译小任务在 20 个句对上 token accuracy 为 0.74%，exact match 为 0，说明 50 步仅跑通评测闭环，尚未达到可用翻译质量。

本次 CUDA 运行实际开启 AMP 和 activation checkpointing；显存不足时降低 micro-batch、增加梯度累积，再降低 block size。完整长程 0.1B 收敛仍需更多 token 和训练步数。

## 下一步是什么

继续扩大维基/小语种/代码数据比例和 token 数，逐步增加 optimizer steps；用更多英法句对和真正的分类/翻译验证集替换当前 20 条小任务，比较不同 tokenizer 与学习率调度下的最佳验证 PPL。
