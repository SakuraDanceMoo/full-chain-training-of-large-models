# 阶段三单卡 CUDA 实验记录

日期与时区：2026-10-11（Asia/Hong_Kong）

## 做了什么

建立 `03_pretraining/` 下的单卡 CUDA 预训练工程。数据管线只读取本阶段的维基摘要、斯瓦希里语和 Python 代码四个文件，未复制或读取英法歌曲；使用阶段一 BPE tokenizer 和 token cache。模型采用 0.1B profile，`batch_size=1`、`grad_accum=4`、`block_size=128`，运行 50 个 optimizer steps，并启用 `--amp --activation-checkpointing`。

## 结果是什么

真实终端输出和指标保存在 `logs/train_stdout.log`、`logs/training_metrics.csv` 和 `logs/training_config.json`。模型参数量为 85,547,520，token 数 8,851，验证 PPL 从 594.85 降到 134.15；独立评测 PPL 为 121.31。最佳权重为 `models/best_checkpoint.json` 及其 9 个分片，曲线为 `curves/training_loss.png`。

## 下一步是什么

继续增加非英法语料的 token 数和训练步数，记录 AMP、梯度累积和激活重计算带来的显存峰值、吞吐和收敛差异；后续评测使用独立分类或翻译任务，但不得把英法歌曲并入本阶段训练语料。
