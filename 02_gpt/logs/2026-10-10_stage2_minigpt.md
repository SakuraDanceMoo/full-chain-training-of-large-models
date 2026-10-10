# 阶段二 Mini-GPT 实验记录

日期与时区：2026-10-10（Asia/Hong_Kong）  
阶段：`02_gpt`  
代码：[`scripts/minigpt.py`](../scripts/minigpt.py)  
数据：[`data/raw/tinyshakespeare.txt`](../data/raw/tinyshakespeare.txt)

## 做了什么

复用 `09_encoder_decoder/scripts/transformer.py` 中 EncoderBlock 的注意力、FFN、残差和 LayerNorm 分解，改写为带严格下三角因果掩码的 `CausalSelfAttention` 和 `GPTBlock`。实现 token embedding、可学习位置 embedding、4 层 GPTBlock、最终 LayerNorm、权重绑定输出投影、带 `target_mask` 的 next-token 交叉熵，以及 temperature/top-k 生成函数。模型共 4 层、4 头、128 维、上下文长度 128；字符词表保证训练文本可逆。

使用 tinyShakespeare UTF-8 原始文本，按 90%/10% 划分训练集和验证集，固定 seed=1337，在 CPU 单设备运行 1200 个优化步骤。

## 结果是什么

真实指标保存在 [`training_metrics.csv`](training_metrics.csv)，曲线在 [`../curves/training_loss.png`](../curves/training_loss.png)：

| step | train loss | validation loss |
|---:|---:|---:|
| 0 | 4.2292 | 4.2283 |
| 600 | 2.2594 | 2.2860 |
| 1200 | 1.9754 | 2.0511 |

loss 持续下降，说明因果 mask、标签右移和优化闭环正常。生成样例见 [`generation_samples.txt`](generation_samples.txt)：以 `ROMEO:` 开头后，模型生成了 `VORILALA`、`DULIA`、`DUKE OLIZART` 等角色格式、换行和部分英语短语，已具备 tinyShakespeare 的表面文体；由于数据量、模型规模和训练步数很小，角色名和语义仍有明显错误，不能当作语言理解能力证据。

checkpoint、配置和汇总分别在 [`../models/minigpt_tinyshakespeare.pt`](../models/minigpt_tinyshakespeare.pt)、[`training_config.json`](training_config.json) 和 [`stage2_summary.json`](stage2_summary.json)。

## 下一步是什么

将字符级输入替换为阶段一训练的 BPE/SentencePiece 词表，比较相同参数预算下的 perplexity；在阶段三加入更大的混合语料、checkpoint 恢复、梯度累积和独立评测集，并记录可复现实验 seed 与硬件信息。
