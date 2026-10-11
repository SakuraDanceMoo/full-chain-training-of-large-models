"""Render the stage-three decoder-only pretraining architecture."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "03_pretraining" / "logs" / "model_architecture.png"
FONT_FILE = Path(r"C:\Windows\Fonts\msyh.ttc")
if FONT_FILE.exists():
    CHINESE_FONT = font_manager.FontProperties(fname=str(FONT_FILE))
else:
    CHINESE_FONT = None


def box(axis, x, y, width, height, title, subtitle, color):
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.018,rounding_size=0.02",
                           linewidth=1.6, edgecolor="#25364d", facecolor=color)
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height * .62, title, ha="center", va="center", fontsize=13, weight="bold", color="#162235", fontproperties=CHINESE_FONT)
    axis.text(x + width / 2, y + height * .30, subtitle, ha="center", va="center", fontsize=10, color="#25364d", fontproperties=CHINESE_FONT)


def arrow(axis, x1, y1, x2, y2):
    axis.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "-|>", "lw": 1.6, "color": "#425466"})


def main() -> None:
    fig, axis = plt.subplots(figsize=(12, 7), dpi=180)
    axis.set_xlim(0, 12); axis.set_ylim(0, 8); axis.axis("off")
    axis.text(6, 7.55, "Stage 3 · Decoder-only 0.1B Pretraining Architecture", ha="center", va="center", fontsize=20, weight="bold", color="#162235", fontproperties=CHINESE_FONT)
    axis.text(6, 7.15, "BPE token IDs → causal GPT blocks → next-token logits", ha="center", va="center", fontsize=11, color="#526273", fontproperties=CHINESE_FONT)
    x, width, height = 3.8, 4.4, .72
    steps = [
        (6.0, "输入 token IDs", "BPE vocabulary · V = 512", "#dbeafe"),
        (5.05, "Token Embedding + Position Embedding", "每个位置映射为 768 维向量", "#d9f2e6"),
        (4.10, "Block 1", "Causal Self-Attention (12 heads × 64d) + FFN + LayerNorm", "#ffe9c7"),
        (3.15, "Block 2", "Causal Self-Attention (12 heads × 64d) + FFN + LayerNorm", "#ffe9c7"),
    ]
    for y, title, subtitle, color in steps:
        box(axis, x, y, width, height, title, subtitle, color)
    axis.text(6, 2.34, "⋮", ha="center", va="center", fontsize=28, color="#425466")
    box(axis, x, 1.42, width, height, "Block 12", "Causal Self-Attention (12 heads × 64d) + FFN + LayerNorm", "#ffe9c7")
    box(axis, x, .48, width, height, "输出 logits", "Final LayerNorm → tied output projection → V logits", "#e8dcff")
    for y in (6.0, 5.05, 4.10, 3.15, 2.14, 1.42):
        arrow(axis, 6, y - .02, 6, y - .23)
    axis.text(10.15, 4.1, "每个 block\n\n12 heads × 64d\n= 768 hidden\n\nFFN: 768 → 3072 → 768\n\ncausal mask: 只能看过去", ha="center", va="center", fontsize=11, color="#25364d", fontproperties=CHINESE_FONT,
              bbox={"boxstyle": "round,pad=.55", "facecolor": "#f5f7fa", "edgecolor": "#b7c4d2"})
    axis.text(1.65, 3.85, "训练目标\nnext-token cross entropy\n\n显存策略\ngradient accumulation\nAMP · activation checkpointing", ha="center", va="center", fontsize=11, color="#25364d", fontproperties=CHINESE_FONT,
              bbox={"boxstyle": "round,pad=.55", "facecolor": "#f5f7fa", "edgecolor": "#b7c4d2"})
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
