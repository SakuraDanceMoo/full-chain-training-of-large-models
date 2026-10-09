# Tokenizer 实验记录

## 做了什么

以 566 对英法歌曲句子复现手写 BPE、官方 SentencePiece Unigram 和五档词表压缩率对照，归档模型与运行日志。

## 结果是什么

完整语料与 7 个边界测试均可逆，未产生未知 token；1,024 词表时留出数据 BPE 为 2.7042 字符/token，Unigram 为 2.2356。

## 下一步是什么

固定 tokenizer 版本，进入 Decoder-only 与 Mini-GPT 的训练/生成闭环。

完整配置、限制和证据见 [统一实验记录](../../docs/experiments/2026-10-09_repository_and_tokenizer.md)。
