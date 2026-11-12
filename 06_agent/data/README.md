# 长程 Agent 与自我评估 数据

`raw/` 保存原始数据，`processed/` 保存派生数据。每次处理记录来源、编码、SHA-256、配置与处理脚本；不覆盖原始数据。

评测任务由 `scripts/agent.py` 按 config seed=20261112 生成，写入 `processed/tasks.json`。任务取三个不重复的 00–31 整数，要求找出最大的两个并求和。当前样本数 24；问题、期望答案与处理逻辑可由脚本复现。RLVR 使用阶段五已登记的训练/验证划分与精确比较 verifier。
