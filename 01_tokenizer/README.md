# 阶段一 从零实现 Tokenizer

语料使用 `data/raw/eng_fra_songs.txt`：从原有 Encoder-Decoder 目录复制归档，包含 566 对英法歌曲句子，分别读取两列得到 1,132 句。原始文件保持不变，不转小写、不折叠空格、不规范化重音字符；校验值见 `data/manifest.json`。

## 实现与文件

- `tokenizer.py`：仅用 Python 标准库手写 Unicode BPE。字符初始化 → 统计相邻 pair → 选最高频 pair → 从左到右非重叠合并 → 重新统计，直到达到词表预算或最低频率门槛。相同频率按 pair 字典序确定选择，使规则稳定。
- `corpus.py`：验证并读取英法两列；使用相对代码文件的位置解析语料路径，运行不依赖当前工作目录。
- `sentencepiece_tokenizer.py`：调用**官方 SentencePiece Unigram**，使用概率词表训练与最优路径分词。SentencePiece 原生支持 BPE/Unigram，不支持 WordPiece，所以这里选择用户要求中的 Unigram 对照。
- `demo.py`：完整歌曲语料上的 BPE 训练与分词可视化，显示 pieces、IDs 和还原文本。
- `evaluate_tokenizers.py`：按句对划分训练/留出数据，生成频率统计、合并日志、词表、示例、压缩率曲线及报告。
- `test_tokenizer.py`：所有歌曲句子的往返验证，另测空串、连续空格、换行、未知中文/emoji、组合重音、特殊 token 字面量、SentencePiece 空格标记冲突，以及保存加载一致性。

## 运行

所有 Python 实现位于 `scripts/`。在本仓库根目录使用已安装 `requirements.txt` 的 Python 环境：

```powershell
$env:PYTHONUTF8 = '1'
python scripts/run_stage1.py
# 或分别运行
python 01_tokenizer/scripts/demo.py
python 01_tokenizer/scripts/test_tokenizer.py
python 01_tokenizer/scripts/evaluate_tokenizers.py
```

依赖记录在本仓库根目录的 `requirements.txt`，包括官方 `sentencepiece` 与 `matplotlib`。BPE 自身无第三方依赖。

## 可逆性

两个模型都保留语料字符，另保留 256 个 UTF-8 字节 token。未见字符按 UTF-8 字节回退，不丢成 `<unk>`。BPE 满足 `decode(encode(text)) == text`，支持有效 Unicode 字符串；插入的 BOS/EOS 在默认解码时跳过。人为输入 `<unk>` ID 或不能构成有效 UTF-8 的 ID 序列，不属于这一保证。

SentencePiece 使用 `identity` 规范化、关闭 dummy prefix 与多余空格删除、开启 byte fallback。SentencePiece 将 `▁` 当作空格标记；封装对包含该字面字符的输入使用官方 byte token 编码，保持原文。两种 tokenizer 的 token ID 不要求相同，交叉核验的是原文还原和统一指标。

词表预算包含 4 个特殊 token、256 个 byte token 和训练字符；小于这些基本项的预算会报错。旧版词表 JSON 与新的字节回退格式不兼容，应重新训练。

## 实验与产出

固定 seed 42，把**句对**按约 80%/20% 分组，同一对中的英语和法语一起分组。歌曲中的重复歌词先合并到同组，防止同一句同时出现在训练与留出数据。两个模型只用训练组学习词表，然后在训练、留出、完整语料分别核验；实际分组数量见报告。这里的交叉验证是两个算法的对照和一次固定留出评估，不是 K 折交叉验证。

默认词表预算为 384、512、640、768、1024。平均每 token 字符数定义为：

`sum(len(sentence)) / sum(len(tokenizer.encode(sentence)))`

字符按 Unicode code point 计数，包含原文空格/标点，不计额外 BOS/EOS。数值越大，同一文本需要的 token 越少。曲线横轴使用**实际词表大小**；图中同时展示训练与留出数据，避免把记住歌曲文本的收益当作泛化收益。

输出分别归档在 `logs/`、`models/` 和 `curves/`：

- `logs/initial_pair_frequencies.csv`：训练语料最初字符 pair 的频率。
- `logs/bpe_*_merge_history.csv`：每一步选择的 pair、当时频率、合并结果及词表大小。
- `models/bpe_*.json`：训练好的 BPE 词表、合并规则与字节回退信息。
- `models/unigram_*.model` / `models/unigram_*.vocab`：官方 SentencePiece 模型与词表。
- `logs/tokenization_examples.json`：同一留出文本的两种分词、ID 与原文还原。
- `logs/tokenizer_compression_results.csv`：分组测量数据和往返失败/未知 token 数。
- `curves/tokenizer_compression_curve.png` / `.svg`：可查看/可放大的压缩率曲线。
- `logs/comparison_report.md`：结果表。
- `logs/experiment_manifest.json`：原始语料 SHA-256、分组 seed、版本、指标与原文未改动核验。
- `logs/*_run.log`、`logs/reproduction_run.json`：本次真实终端输出、命令、时间与退出状态。

这份小语料实验比较分词压缩率，不能据此判断哪一种能提升下游语言模型质量。它是教学实现，频率重计数与逐规则编码适合本语料，不适合直接处理工业规模训练数据。

## 与旧 Vocab 的区别

`09_encoder_decoder/scripts/machine_translation.py` 中的 `Vocab` 接收**已经切好的 token 列表**，按出现频率过滤并分配 ID；它本身不学习分词规则。当前旧流程的 `tokenize_nmt` 按空格和标点产生词级 token，再交给 `Vocab`。

| 项目 | 旧 Vocab 与 tokenize_nmt | 新 BPE Tokenizer |
| --- | --- | --- |
| 输入 | 已分好的词/标点列表 | 原始文本 |
| 学习内容 | token 频率和 ID 映射 | 字符 pair 合并规则、子词词表和 ID 映射 |
| 低频/新词 | 低频词被过滤，新词映射到 `<unk>` | 可拆为已知字符/子词；新字符按字节回退 |
| 空格/原文 | `split()` 丢失空格格式，无法精确复原 | 保留空格/标点与原文，encode/decode 可逆 |
| 词表大小 | 由词频阈值与词类型数决定 | 由词表预算约束，包含基本字符与回退项 |
| 保存 | 原类未提供序列化接口 | JSON 保存/加载词表及合并顺序 |
| 当前用途 | Seq2Seq 英法翻译的源/目标词表 | 后续 Mini-GPT 与预训练的文本编码器 |

Tokenizer 内部也有类似 Vocab 的双向 ID 表，但比 Vocab 多了“如何把原文切成 token”这一层。`playing` 不在旧词表时会变为一个 `<unk>`；新 tokenizer 能将它拆成可表示的片段（实际边界由训练规则决定）。

旧 Vocab 中 `<unk>` 为 0；本阶段约定 `<pad>=0`、`<unk>=1`、`<bos>=2`、`<eos>=3`，不能把新 ID 直接送入依赖旧词表的 checkpoint。接入后续模型时应按新词表大小重建 embedding/输出层，并重新编码数据。
