# Decoder-only 与 Mini-GPT

状态：**已完成（implemented）**。

实现因果掩码、next-token loss、采样与可学习位置编码；在 tinyShakespeare 上验证 loss 下降并生成样例。

## 目录约定

- `scripts/`：核心实现、训练、评测与阶段验收脚本。
- `data/raw/`：原始 tinyShakespeare 文本及来源清单；`data/processed/`：派生数据。
- `logs/`：训练配置、指标 CSV、checkpoint 说明和生成样例。
- `curves/`：由真实指标生成的 `training_loss.png`。
- `models/`：本次训练得到的 `minigpt_tinyshakespeare.pt`，可用配置和词表复现生成。

## 验收

在仓库根目录执行：

```powershell
python 02_gpt/scripts/check_stage.py
```

阶段验收脚本会检查核心代码、训练指标和 loss 曲线。

预期交付：

- `scripts/minigpt.py`
- `logs/training_metrics.csv`
- `curves/training_loss.png`

## 复现

在仓库根目录运行（不依赖 `transformers`）：

```powershell
python 02_gpt/scripts/minigpt.py --steps 1200 --eval-interval 200 --device cpu
```

`--data` 可替换为任意 UTF-8 纯文本，例如 TinyStories；脚本会按字符建立可逆词表。默认配置为 4 层、4 头、128 维、上下文 128，适合单卡或 CPU 小规模验证。

## 实现要点

- `CausalSelfAttention` 使用下三角 mask，位置 `t` 的 query 永远看不到 `t+1` 之后的 token。
- `GPTBlock` 沿用已有 `EncoderBlock` 的“注意力 + FFN + 残差/LayerNorm”分解，使用 pre-LN 形式以便堆叠。
- `GPT` 由 token embedding、可学习 position embedding、N 个 block、最终 LayerNorm 和权重绑定的输出投影组成。
- `next_token_loss` 支持 `target_mask`，可在有 padding 的 batch 中只对有效标签计算交叉熵。
- `generate` 支持 temperature、top-k 和贪心/采样生成；上下文超过 block size 时自动截断到最近窗口。

训练记录见 [`logs/2026-10-10_stage2_minigpt.md`](logs/2026-10-10_stage2_minigpt.md)，生成样例见 [`logs/generation_samples.txt`](logs/generation_samples.txt)。

下一阶段：03_pretraining。
