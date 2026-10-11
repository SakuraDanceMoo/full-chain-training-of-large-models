# 阶段三数据

`raw/` 保存本阶段原始语料，`processed/` 保存由训练脚本生成的 token cache。原始文件只读归档，来源、编码和 SHA-256 记录在 `raw/manifest.json`。

本阶段语料边界为维基摘要、斯瓦希里语和 Python 代码；英法歌曲只在清单中标记为排除项，不参与 token stream。
