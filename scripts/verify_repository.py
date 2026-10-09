"""Validate stage layout, syntax, raw-data hashes and declared artifact gates."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("scripts", "data/raw", "data/processed", "logs", "curves")


def describe_stage(stage: Path) -> dict:
    config = json.loads((stage / "stage.json").read_text(encoding="utf-8"))
    missing_layout = [name for name in REQUIRED if not (stage / name).is_dir()]
    if not (stage / "README.md").is_file():
        missing_layout.append("README.md")
    missing_artifacts = [name for name in config["acceptance_files"] if not (stage / name).is_file()]
    return {"stage": stage.name, "declared_status": config["status"],
            "layout_ok": not missing_layout, "missing_layout": missing_layout,
            "missing_acceptance_artifacts": missing_artifacts,
            "artifact_gate_passed": not missing_artifacts}


def check_stage(stage: Path) -> int:
    result = describe_stage(stage)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["layout_ok"] and result["artifact_gate_passed"] else 2


def main() -> None:
    registry = json.loads((ROOT / "stages.json").read_text(encoding="utf-8"))
    stages = [describe_stage(ROOT / entry["directory"]) for entry in registry]
    checked_files = []
    for path in ROOT.rglob("*.py"):
        if any(part in (".git", ".venv", "__pycache__") for part in path.parts):
            continue
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        checked_files.append(path.relative_to(ROOT).as_posix())
    hashes = []
    for manifest in ROOT.glob("*/data/manifest.json"):
        stage = manifest.parent.parent
        for entry in json.loads(manifest.read_text(encoding="utf-8"))["files"]:
            data_path = stage / entry["path"]
            actual = hashlib.sha256(data_path.read_bytes()).hexdigest()
            hashes.append({"path": data_path.relative_to(ROOT).as_posix(),
                           "sha256": actual, "matches_manifest": actual == entry["sha256"]})
    passed = all(item["layout_ok"] for item in stages) and all(item["matches_manifest"] for item in hashes)
    passed = passed and all(item["artifact_gate_passed"] for item in stages if item["declared_status"] != "planned")
    result = {"repository_structure_and_hash_checks_passed": passed,
              "stages": stages, "python_syntax_checked": checked_files, "raw_data_hash_checks": hashes,
              "note": "Planned stages have explicit missing deliverables; layout readiness is not experiment completion."}
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "repository_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Structure/hash checks: {'PASS' if passed else 'FAIL'}; Python syntax: {len(checked_files)} files")
    for item in stages:
        print(f"{item['stage']}: {item['declared_status']}, layout={item['layout_ok']}, missing deliverables={len(item['missing_acceptance_artifacts'])}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
