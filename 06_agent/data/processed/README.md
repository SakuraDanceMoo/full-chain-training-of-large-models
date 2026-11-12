# 派生数据

`tasks.json` 为 24 道确定性生成的评测任务，seed 记录于 `../config.json`。它用于 Agent 端到端评测，不参与 RLVR 参数更新。策略训练复用阶段五拆分好的整数比较训练集，避免把 Agent 测试答案用于训练。
