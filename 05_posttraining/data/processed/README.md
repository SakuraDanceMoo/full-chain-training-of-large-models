# 派生数据

`train.json`、`validation.json`、`test.json` 由 `scripts/data_pipeline.py` 从 raw 确定性分层划分，分别为 768、128、128 条。题目 ID 不跨 split；`manifest.json` 记录每份文件的 SHA-256。派生 JSON 固定输出 UTF-8 LF，便于跨平台检验。
