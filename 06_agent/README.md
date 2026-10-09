# 长程 Agent 与自我评估

状态：**待开展（planned）**。

建立可验证任务环境与 harness，记录工具轨迹、反思及任务成功率。

## 目录约定

- `scripts/`：核心实现、训练、评测与阶段验收脚本。
- `data/raw/`：原始数据；`data/processed/`：派生数据。
- `logs/`：运行输出、配置、指标和实验记录。
- `curves/`：由真实指标生成的实验曲线；当前尚未产生。

## 验收

在仓库根目录执行：

```powershell
python 06_agent/scripts/check_stage.py
```

尚未具备核心产物时返回退出码 2，并列出缺项。目录初始化不代表本阶段实现完成。

预期交付：

- `scripts/agent.py`
- `scripts/evaluate.py`
- `logs/trajectories.jsonl`
- `curves/success_rate.png`

下一阶段：汇编大项目报告。
