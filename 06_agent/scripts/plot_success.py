"""Plot cumulative success rate from harness trajectories."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "06_agent"
rows = [json.loads(line) for line in (STAGE / "logs/trajectories.jsonl").read_text(encoding="utf-8").splitlines() if line]
running_final = [sum(row["success"] for row in rows[:i]) / i for i in range(1, len(rows) + 1)]
running_first = [sum(row["first_pass_success"] for row in rows[:i]) / i for i in range(1, len(rows) + 1)]
out = STAGE / "curves/success_rate.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(8, 4.5))
plt.plot(range(1, len(running_first) + 1), [100 * x for x in running_first], marker="o", linewidth=1.5,
         label="Before verifier reflection")
plt.plot(range(1, len(running_final) + 1), [100 * x for x in running_final], marker="o", linewidth=1.8,
         label="After verifier reflection")
plt.ylim(0, 105)
plt.xlabel("Completed tasks")
plt.ylabel("Cumulative task success (%)")
plt.title("Stage 6 verifier-backed agent evaluation")
plt.legend()
plt.grid(alpha=.25)
plt.tight_layout()
plt.savefig(out, dpi=160)
print(out)
