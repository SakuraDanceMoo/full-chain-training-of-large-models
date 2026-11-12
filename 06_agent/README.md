# 长程 Agent 与自我评估

状态：**已实现并实测（范围受限）**。本阶段在阶段五的共享 SFT 预热基座上，以精确比较奖励做 24 步 RLVR 微调，再将 LoRA 策略接入多步数值任务 harness。任务分解和工具路由由固定模板负责，实验验证阶段五比较能力在工具流程中的使用与自我纠错，不代表通用长程规划能力。

## 目录约定

- `scripts/`：核心实现、训练、评测与阶段验收脚本。
- `data/raw/`：原始数据；`data/processed/`：派生数据。
- `logs/`：运行输出、配置、指标和实验记录。
- `curves/`：由真实指标生成的训练和成功率曲线。
- `models/`：阶段五共享预热基座上的阶段六 RLVR LoRA 增量。

## 验收

在仓库根目录执行以下命令，按顺序微调策略、运行任务、统计指标并画图：

```powershell
..\.venv\Scripts\python.exe 06_agent/scripts/train_policy.py
..\.venv\Scripts\python.exe 06_agent/scripts/agent.py
..\.venv\Scripts\python.exe 06_agent/scripts/evaluate.py
..\.venv\Scripts\python.exe 06_agent/scripts/plot_success.py
..\.venv\Scripts\python.exe 06_agent/scripts/plot_training.py
..\.venv\Scripts\python.exe 06_agent/scripts/check_stage.py
```

复现需使用仓库根目录的 Python 环境，并依赖阶段三、五已归档权重。

预期交付：

- `scripts/train_policy.py`：以组内精确比较奖励更新策略。
- `scripts/agent.py`：工具调用、分解执行、判分与反思日志。
- `scripts/evaluate.py`：统计首答/纠错后成功率、子任务失败率与工具切换正确率。
- `data/processed/tasks.json`、`logs/trajectories.jsonl`、`logs/metrics.json`：固定测试集及运行证据。
- `models/agent_rlvr_lora_s20261112.pt`：24 步 RLVR 微调增量。
- `curves/success_rate.png`、`curves/rlvr_reward.png`：真实运行曲线。

本次 RTX 4070 Ti 实测：首答成功率 70.83%，判分与反思纠错后完成率 100%，比较子任务准确率 86.11%，固定模板工具路由正确率 100%，24 个任务有 7 个触发反思。工具路由由模板确定，100% 只适用于该任务格式。详见 [`logs/2026-11-12_stage6_agent.md`](logs/2026-11-12_stage6_agent.md)。

下一阶段：汇编大项目报告。
