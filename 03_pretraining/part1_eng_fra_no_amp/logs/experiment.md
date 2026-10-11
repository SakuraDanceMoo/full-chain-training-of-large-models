# 阶段三第一部分实验记录

日期与时区：2026-10-11（Asia/Hong_Kong）

## 做了什么

建立独立的英法歌曲预训练工程。数据管线只读取 `data/raw/eng_fra_songs.txt`，使用阶段一 BPE tokenizer 生成 token stream，按 90%/10% 划分训练和验证；复用 GPT 的 0.1B profile，在单卡 CUDA 上训练 20 steps。按照前半部基线要求显式关闭 AMP 和 activation checkpointing。

## 结果是什么

真实终端输出和指标保存在 `logs/training_metrics.csv`、`logs/training_config.json`。模型参数量为 85,547,520，token 数 23,978，验证 PPL 从 648.66 降到 225.91；独立评测 PPL 为 200.90。最佳权重为 `models/best_checkpoint.json` 及其 9 个分片，曲线为 `curves/training_loss.png`。

## 下一步是什么

在保持英法语料隔离的前提下增加 token 数和 optimizer steps，比较 no-AMP 与 AMP 的吞吐、显存和收敛；不要把第二部分的混合语料合入本实验。
