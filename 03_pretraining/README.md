# 0.1B 端到端预训练

状态：**待开展（planned）**。

完成数据清洗、混合精度/梯度累积、checkpoint、PPL 与复现配置。

## 目录约定

- `scripts/`：核心实现、训练、评测与阶段验收脚本。
- `data/raw/`：原始数据；`data/processed/`：派生数据。
- `logs/`：运行输出、配置、指标和实验记录。
- `curves/`：由真实指标生成的实验曲线；当前尚未产生。

## 验收

在仓库根目录执行：

```powershell
python 03_pretraining/scripts/check_stage.py
```

尚未具备核心产物时返回退出码 2，并列出缺项。目录初始化不代表本阶段实现完成。

预期交付：

- `scripts/train.py`
- `scripts/evaluate.py`
- `logs/training_metrics.csv`
- `curves/training_loss.png`

下一阶段：04_systems_scaling。
