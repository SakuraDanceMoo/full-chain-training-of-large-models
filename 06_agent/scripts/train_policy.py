"""Fine-tune the agent comparison policy with exact verifier rewards."""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "06_agent"
POST = ROOT / "05_posttraining"
sys.path.insert(0, str(POST / "scripts"))
import engine


def main() -> None:
    stage_config = json.loads((STAGE / "config.json").read_text(encoding="utf-8"))
    post_config = json.loads((POST / "config.json").read_text(encoding="utf-8"))
    config = {**post_config, **stage_config}
    config["steps"] = stage_config["rlvr_steps"]
    config["eval_interval"] = stage_config["eval_interval"]
    # Reuse the audited stage-five RLVR objective and data verifier, while keeping
    # this run's checkpoint, trajectory, and metrics inside stage six.
    engine.STAGE = STAGE
    result = engine.train(config, "rlvr", "lora", stage_config["seed"],
                          output_id="agent_rlvr_lora_s20261112")
    checkpoint = STAGE / result["checkpoint"]
    manifest = {
        "checkpoint": checkpoint.name,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "bytes": checkpoint.stat().st_size,
        "base_manifest_sha256": hashlib.sha256((ROOT / stage_config["base_checkpoint"]).read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256((ROOT / stage_config["tokenizer"]).read_bytes()).hexdigest(),
        "run_id": result["run_id"],
        "method": "stage-five GRPO-style RLVR objective with exact comparison verifier",
        "limitations": "reward/task family is binary integer comparison; task decomposition and tool routing remain deterministic",
    }
    (STAGE / "models/manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
