# 混合预训练语料

本目录用于阶段三的增量实验，训练脚本会按文件名排序读取 `.txt`、`.py`、`.md` 文件，逐文件追加 `<eos>`。

| 文件 | 类型 | 来源 / 用途 | SHA-256 |
|---|---|---|---|
| `multilingual_eng_fra_songs.txt` | 英文/法文 | 阶段九原始 `eng_fra_songs.txt`；多语种/低资源代理 | `4d927cb4d2bb2ebbdba1b9914175c05a308a37c5bd0d9ea5b39a03aabafd6ea3` |
| `wiki_transformer.txt` | 英文维基 | Wikipedia Transformer (deep learning) 摘要，CC BY-SA 内容 | `7578384f7a4e7ec4ee0d63a7e8ee1c01b6245e554127c36a0d5c9ae548adc5b6` |
| `wiki_language_model.txt` | 英文维基 | Wikipedia Language model 摘要，CC BY-SA 内容 | `70018e95ac9addd00441e56981e73530eca198bcf15dd1f9ca71f86c354864be` |
| `swahili_language.txt` | 斯瓦希里语 | Swahili Wikipedia Lugha 摘要，小语种样例 | `b83907fa398ab8de72402bd89f0a1a199c15c5bf84fe81ad9af057c8f93ed24f` |
| `code_python_subset.py` | Python 代码 | 本仓库阶段二/三训练代码子集，保留源码用于代码分布样例 | `c631439413a87440839b864980dc70b129eb9944216ac8293ff1091c48079cce` |

Wikipedia 摘要获取日期：2026-10-11（Asia/Hong_Kong）。完整 URL 和文件大小见同目录 `manifest.json`；该清单不作为训练文本读取。
