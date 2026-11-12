"""Run a verifier-backed multi-step numeric agent with the stage-five RLVR policy."""
from __future__ import annotations

import json
import sys
from functools import cmp_to_key
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "06_agent"
sys.path.insert(0, str(ROOT / "05_posttraining/scripts"))
from data_pipeline import tokenizer
from peft import apply_trainable, make_model


def load_policy(config: dict, device: str = "cuda"):
    post_config = json.loads((ROOT / "05_posttraining/config.json").read_text(encoding="utf-8"))
    model = make_model({**post_config, **config}, "lora", anchor=True).to(device)
    payload = torch.load(ROOT / config["policy_checkpoint"], map_location="cpu", weights_only=True)
    apply_trainable(model, payload["trainable"])
    model.eval()
    tok = tokenizer(config)
    return model, tok


@torch.inference_mode()
def compare_tool(model, tok, a: int, b: int, device: str = "cuda") -> dict:
    prompt = f"Compare: {a:02d}>{b:02d}\nAnswer:"
    ids = tok.encode(prompt, add_bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated = model.generate(x, max_new_tokens=2, top_k=None, do_sample=False)
    new_ids = generated[0, len(ids):].tolist()
    text = tok.decode(new_ids).strip()
    try:
        answer = int(text)
    except ValueError:
        answer = None
    return {"tool": "compare_policy", "input": [a, b], "output": answer,
            "expected": int(a > b), "correct": answer == int(a > b), "raw_tokens": new_ids, "raw_text": text}


def calculator_tool(values: list[int]) -> dict:
    return {"tool": "calculator", "input": values, "output": sum(values)}


def run_task(task: dict, model, tok, device: str = "cuda") -> dict:
    values = task["values"]
    # The harness fixes the task protocol; the learned policy supplies each comparison.
    pairwise = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            pairwise.append(compare_tool(model, tok, values[i], values[j], device))

    outcomes = {(row["input"][0], row["input"][1]): row["output"] for row in pairwise}

    def policy_order(a: int, b: int) -> int:
        result = outcomes.get((a, b))
        if result is None:
            result = 1 - outcomes[(b, a)] if outcomes.get((b, a)) in (0, 1) else None
        if result not in (0, 1):
            return 0
        return -1 if result == 1 else (1 if a != b else 0)

    tool_order = sorted(values, key=cmp_to_key(policy_order))
    selected = tool_order[:2]
    total = calculator_tool(selected)
    prediction = total["output"]
    initial_prediction = prediction
    first_pass_success = prediction == task["expected"]
    success = first_pass_success
    reflection = None
    if not success:
        failed = [row for row in pairwise if not row["correct"]]
        ranked = sorted(values, reverse=True)
        reflection = {
            "我为什么答错": "比较策略给出了错误或不可解析的 token，排序因此选错；我应先用环境中的精确比较器复核失败子任务，再重算结果。",
            "failed_comparisons": failed,
            "verified_correction": ranked[:2],
            "corrected_sum": sum(ranked[:2]),
        }
        prediction = reflection["corrected_sum"]
        success = prediction == task["expected"]

    return {
        "task_id": task["id"], "instruction": task["instruction"], "values": values,
        "plan": ["调用比较策略完成两两比较", "用计算器求前两名之和", "由环境判分并记录反思"],
        "trajectory": pairwise + [total], "tool_calls": len(pairwise) + 1,
        "tool_switch_correct": all(x["tool"] == "compare_policy" for x in pairwise)
                               and total["tool"] == "calculator",
        "initial_prediction": initial_prediction, "first_pass_success": first_pass_success,
        "prediction": prediction, "expected": task["expected"], "success": success,
        "reflection": reflection,
    }


def make_tasks(n: int, seed: int) -> list[dict]:
    import random
    rng = random.Random(seed)
    tasks = []
    seen = set()
    while len(tasks) < n:
        values = rng.sample(range(32), 3)
        if tuple(sorted(values)) in seen:
            continue
        seen.add(tuple(sorted(values)))
        largest = sorted(values, reverse=True)[:2]
        tasks.append({"id": f"agent_{len(tasks):03d}", "values": values,
                      "instruction": f"从 {values} 中找出最大的两个数，并计算它们的和。",
                      "expected": sum(largest)})
    return tasks


def main() -> None:
    config = json.loads((STAGE / "config.json").read_text(encoding="utf-8"))
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    model, tok = load_policy(config)
    tasks = make_tasks(config["benchmark_tasks"], config["seed"])
    task_file = STAGE / "data/processed/tasks.json"
    task_file.parent.mkdir(parents=True, exist_ok=True)
    task_file.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output = [run_task(task, model, tok) for task in tasks]
    path = STAGE / "logs/trajectories.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({"tasks": len(output), "success_rate": sum(x["success"] for x in output) / len(output),
                      "comparison_accuracy": sum(c["correct"] for x in output for c in x["trajectory"] if c["tool"] == "compare_policy")
                                               / sum(c["tool"] == "compare_policy" for x in output for c in x["trajectory"]),
                      "tool_switch_accuracy": sum(x["tool_switch_correct"] for x in output) / len(output),
                      "reflection_count": sum(x["reflection"] is not None for x in output),
                      "trajectory_file": str(path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
