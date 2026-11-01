# 后训练与对齐 曲线

由 `scripts/plot_results.py` 从 `logs/runs/*/training.jsonl` 和 `result.json` 生成真实曲线。

- `alignment_comparison.png` 为三 seed 平均测试准确率与样本标准差。
- `peft_resources.png` 比较同一 SFT 数据与更新预算的训练显存、时间。
- `training_loss.png` 展示各自目标的实际 loss，绝对数值不能跨目标比较。
- `validation_accuracy.png` 为固定验证集的监控结果，不用于选 checkpoint。
