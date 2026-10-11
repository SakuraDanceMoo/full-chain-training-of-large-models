# 阶段三训练日志

- `training_metrics.csv`：每个评估点的 train/validation loss、PPL 和 next-token accuracy。
- `training_config.json`：设备、模型、seed、AMP、梯度累积和 activation checkpointing 配置。
- `train_stdout.log`：实际训练命令和终端指标。
- `evaluation_ppl.json`：独立 held-out PPL 结果。
- `experiment.md`：做了什么、结果是什么、下一步是什么。
- `model_architecture.png`：12 层、12 heads、768 hidden 的模型架构图。
