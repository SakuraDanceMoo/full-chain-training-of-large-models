"""Reproduce stage one and retain each command's actual stdout and exit code."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HK = timezone(timedelta(hours=8))


def main() -> None:
    stage = ROOT / "01_tokenizer"
    logs = stage / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    started = datetime.now(HK).isoformat()
    steps = []
    for script, log_name in (("test_tokenizer.py", "test_run.log"), ("demo.py", "demo_run.log"),
                             ("evaluate_tokenizers.py", "evaluation_run.log")):
        command = [sys.executable, str(stage / "scripts" / script)]
        result = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONUTF8": "1"},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8")
        (logs / log_name).write_text(result.stdout, encoding="utf-8")
        print(result.stdout, end="", flush=True)
        steps.append({"command": f"python 01_tokenizer/scripts/{script}",
                      "exit_code": result.returncode, "log": f"01_tokenizer/logs/{log_name}"})
        if result.returncode:
            break
    manifest = {"started_at": started, "finished_at": datetime.now(HK).isoformat(),
                "timezone": "Asia/Hong_Kong", "steps": steps,
                "passed": len(steps) == 3 and all(step["exit_code"] == 0 for step in steps)}
    (logs / "reproduction_run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
