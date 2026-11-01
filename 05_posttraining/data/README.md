# 后训练与对齐 数据

`raw/` 保存原始数据，`processed/` 保存派生数据。每次处理记录来源、编码、SHA-256、配置与处理脚本；不覆盖原始数据。

当前为 `00..31` 整数比较，1024 条本地穷举数据，按 CC0-1.0 提供。`scripts/data_pipeline.py` 根据固定 seed 按标签分层并按题目隔离，训练 768、验证 128、测试 128。来源与原始字节 SHA-256 见 `manifest.json`。

所有后训练和 PEFT 方案共用这份数据。正确回复构造 SFT 目标及 DPO chosen，相反回复构造 DPO rejected；RLVR 的奖励来自实际生成的精确验分，仅用 train 题目更新。
