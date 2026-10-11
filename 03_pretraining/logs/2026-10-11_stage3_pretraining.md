# 阶段三端到端预训练实验记录

日期与时区：2026-10-11（Asia/Hong_Kong）  
阶段：`03_pretraining`  
架构图：[`model_architecture.png`](model_architecture.png)

## 做了什么

复用阶段一 `BPETokenizer` 和阶段二 `GPT/GPTBlock`，在 `03_pretraining/scripts/` 下实现数据管线、单卡预训练循环、PPL 评测和 checkpoint 保存。训练语料只读取 `data/raw/` 中的维基摘要、斯瓦希里语和 Python 代码，英法歌曲没有进入 token stream。训练启用梯度累积、CUDA AMP 和 activation checkpointing。

本次运行使用 12 层、12 heads、768 hidden、上下文 128、batch 1、梯度累积 4，共 50 个 optimizer steps：

```powershell
python 03_pretraining/scripts/train.py --profile 0.1b --steps 50 --eval-interval 10 --eval-iters 2 --device cuda --amp --activation-checkpointing --block-size 128 --batch-size 1 --grad-accum 4
```

## 结果是什么

指标保存在 [`training_metrics.csv`](training_metrics.csv)，唯一的阶段三 loss 曲线保存在 [`../curves/training_loss.png`](../curves/training_loss.png)：

| step | train loss | validation loss | validation PPL |
|---:|---:|---:|---:|
| 0 | 6.3539 | 6.3883 | 594.85 |
| 30 | 5.0705 | 4.9617 | 142.83 |
| 50 | 5.0391 | 4.8990 | 134.15 |

最佳权重保存为 `models/best_checkpoint.json` 和 9 个分片；独立评测 PPL 为 121.31。

## 下一步是什么

逐步增加非英法语料的 token 数和训练步数，记录显存峰值、吞吐和收敛差异；再用独立分类或翻译小任务替换 held-out next-token proxy。
