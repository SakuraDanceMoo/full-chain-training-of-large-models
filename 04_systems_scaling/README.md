# 系统优化与 Scaling Law

状态：**待开展（planned）**。

实现 Triton 注意力正确性/性能对照，多卡训练与 L(N)/L(D) 拟合。

## 目录约定

- `scripts/`：核心实现、训练、评测与阶段验收脚本。
- `data/raw/`：原始数据；`data/processed/`：派生数据。
- `logs/`：运行输出、配置、指标和实验记录。
- `curves/`：由真实指标生成的实验曲线；当前尚未产生。

## 验收

在仓库根目录执行：

```powershell
python 04_systems_scaling/scripts/check_stage.py
```

尚未具备核心产物时返回退出码 2，并列出缺项。目录初始化不代表本阶段实现完成。

预期交付：

- `scripts/attention_kernel.py`
- `scripts/benchmark.py`
- `scripts/fit_scaling.py`
- `logs/benchmark_metrics.csv`
- `curves/scaling_law.png`

下一阶段：05_posttraining。
