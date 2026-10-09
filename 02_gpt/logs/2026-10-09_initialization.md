# Decoder-only 与 Mini-GPT 初始化记录

日期：2026-10-09（Asia/Hong_Kong）。

## 做了什么

建立 scripts、data/raw、data/processed、logs、curves 及阶段验收入口。

## 结果是什么

目录可追踪，阶段状态为 planned；尚未实现核心训练脚本，也没有模型结果和实验曲线。check_stage.py 是产物验收工具，不是训练实现。

## 下一步是什么

实现因果掩码、next-token loss、采样与 KV Cache；小语料上验证 loss 下降。

验收所需文件：

- `scripts/minigpt.py`
- `logs/training_metrics.csv`
- `curves/training_loss.png`
