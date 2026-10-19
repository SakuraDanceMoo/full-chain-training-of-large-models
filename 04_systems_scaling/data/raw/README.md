# 原始数据

`tinyshakespeare.txt` 原样复制自 `02_gpt/data/raw/tinyshakespeare.txt`，UTF-8。

上游 https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

SHA-256：`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`。

只做分词和派生子集，不改写原文。清单见 `../manifest.json`；Git 通过 `**/data/raw/** -text` 保留原字节。运行前与阶段二原文再次核对。
