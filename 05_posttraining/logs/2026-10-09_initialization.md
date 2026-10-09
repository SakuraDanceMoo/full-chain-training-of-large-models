# 后训练与对齐 初始化记录

日期：2026-10-09（Asia/Hong_Kong）。

## 做了什么

建立 scripts、data/raw、data/processed、logs、curves 及阶段验收入口。

## 结果是什么

目录可追踪，阶段状态为 planned；尚未实现核心训练脚本，也没有模型结果和实验曲线。check_stage.py 是产物验收工具，不是训练实现。

## 下一步是什么

固定同一基座与评测，对照 SFT/DPO/RLVR、LoRA/QLoRA/Adapter。

验收所需文件：

- `scripts/train_sft.py`
- `scripts/train_dpo.py`
- `scripts/train_rlvr.py`
- `logs/evaluation_metrics.csv`
- `curves/alignment_comparison.png`
