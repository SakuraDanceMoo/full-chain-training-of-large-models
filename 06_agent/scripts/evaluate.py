"""Evaluate task success, subtask failures, and tool routing from real trajectories."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "06_agent"


def evaluate(path: Path = STAGE / "logs/trajectories.jsonl") -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    comparisons = [call for row in rows for call in row["trajectory"] if call["tool"] == "compare_policy"]
    summary = {
        "tasks": len(rows),
        "first_pass_success_rate": sum(row["first_pass_success"] for row in rows) / max(1, len(rows)),
        "end_to_end_success_rate": sum(row["success"] for row in rows) / max(1, len(rows)),
        "subtask_failure_rate": sum(not call["correct"] for call in comparisons) / max(1, len(comparisons)),
        "comparison_tool_accuracy": sum(call["correct"] for call in comparisons) / max(1, len(comparisons)),
        "tool_switch_accuracy": sum(row["tool_switch_correct"] for row in rows) / max(1, len(rows)),
        "mean_tool_calls": statistics.mean(row["tool_calls"] for row in rows) if rows else 0,
        "reflections": sum(row["reflection"] is not None for row in rows),
        "trajectory_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
    }
    output = STAGE / "logs/metrics.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
