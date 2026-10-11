"""Check the isolated mixed-corpus AMP experiment."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
required = [
    ROOT / "config.json", ROOT / "data/raw/manifest.json",
    ROOT / "scripts/train.py", ROOT / "scripts/data_pipeline.py",
    ROOT / "logs/training_metrics.csv", ROOT / "logs/training_config.json",
    ROOT / "logs/evaluation_ppl.json", ROOT / "curves/training_loss.png",
    ROOT / "models/best_checkpoint.json",
]
missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
train = json.loads((ROOT / "logs/training_config.json").read_text(encoding="utf-8"))
manifest = json.loads((ROOT / "data/raw/manifest.json").read_text(encoding="utf-8"))
paths = {item["path"] for item in manifest["files"]}
if "eng_fra_songs.txt" in paths or "multilingual_eng_fra_songs.txt" in paths:
    raise SystemExit("cuda_amp_checkpointing corpus must exclude English-French songs")
if not (config["amp"] and config["activation_checkpointing"] and train["amp_enabled"] and train["activation_checkpointing"]):
    raise SystemExit("cuda_amp_checkpointing must enable AMP and activation checkpointing")
if missing:
    raise SystemExit("missing: " + ", ".join(missing))
print("cuda_amp_checkpointing reproducibility checks passed")
