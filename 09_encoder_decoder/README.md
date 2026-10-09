# Encoder-Decoder 基础代码归档

保留原有机器翻译、Seq2Seq、注意力、Transformer、BERT 实现，作为后续阶段的知识基础。本目录保留 09 编号以对应已有学习章节，它不是大模型路线的第九阶段。

## 结构

- `scripts/`：`machine_translation.py`、`encoder_decoder.py`、`seq2seq.py`、`attention_mechanism.py`、`transformer.py`、`bert.py`。
- `data/raw/`：原有 `eng_fra.txt`、`eng_fra_songs.txt`，内容与迁移前一致，见 `data/manifest.json`。
- `data/processed/`：派生数据说明。
- `logs/`：本次导入检查记录、历史产物说明。
- `curves/`：原 `diagram/` 中三张历史 loss 图。

## 运行

在仓库根目录：

```powershell
python 09_encoder_decoder/scripts/machine_translation.py
python 09_encoder_decoder/scripts/seq2seq.py
```

路径根据脚本位置解析，支持不同克隆目录。`seq2seq.py` 默认运行 800 epoch，并会生成 transformer_loss.png；本次归档只验证语法/导入，不重新执行这项训练。

历史曲线缺少可对应的原始训练日志、seed 与环境，标记为继承产物，不声称本次重现。下一步先完成阶段二 Decoder-only 与 Mini-GPT。
