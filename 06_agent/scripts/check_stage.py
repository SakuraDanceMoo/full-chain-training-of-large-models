"""Check stage-six artifacts and consistency of the recorded task metrics."""
import json
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
REQUIRED = [
    "config.json", "scripts/agent.py", "scripts/evaluate.py", "scripts/train_policy.py",
    "data/processed/tasks.json", "models/agent_rlvr_lora_s20261112.pt", "models/manifest.json",
    "logs/trajectories.jsonl", "logs/metrics.json",
    "logs/runs/agent_rlvr_lora_s20261112/training.jsonl",
    "curves/success_rate.png", "curves/rlvr_reward.png",
]


def main() -> int:
    missing = [item for item in REQUIRED if not (STAGE / item).is_file()]
    if missing:
        print("Missing stage-six artifacts:", *missing, sep="\n- ")
        return 2
    tasks = json.loads((STAGE / "data/processed/tasks.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (STAGE / "logs/trajectories.jsonl").read_text(encoding="utf-8").splitlines() if line]
    metrics = json.loads((STAGE / "logs/metrics.json").read_text(encoding="utf-8"))
    if len(tasks) != len(rows) or len(rows) != metrics["tasks"]:
        print("Task, trajectory, and metric counts disagree")
        return 1
    if any("我为什么答错" not in row["reflection"] for row in rows if row["reflection"] is not None):
        print("A reflection record is missing its required explanation")
        return 1
    print(f"Stage six artifacts verified: {len(rows)} tasks, success={metrics['end_to_end_success_rate']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
