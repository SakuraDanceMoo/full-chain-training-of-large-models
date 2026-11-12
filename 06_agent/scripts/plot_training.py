"""Plot mean exact-verifier reward recorded during RLVR policy training."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "06_agent"
source = STAGE / "logs/runs/agent_rlvr_lora_s20261112/training.jsonl"
rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
out = STAGE / "curves/rlvr_reward.png"
plt.figure(figsize=(8, 4.5))
plt.plot([row["step"] for row in rows], [row["rollout_reward"] * 100 for row in rows], marker=".", linewidth=1.5)
plt.xlabel("Optimizer step")
plt.ylabel("Mean exact-match reward (%)")
plt.title("Stage 6 RLVR training reward")
plt.grid(alpha=.25)
plt.tight_layout()
plt.savefig(out, dpi=160)
print(out)
