"""Check the isolated English-French no-AMP experiment."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
required = [
    ROOT / "config.json", ROOT / "data/raw/eng_fra_songs.txt",
    ROOT / "data/raw/manifest.json", ROOT / "scripts/train.py",
    ROOT / "scripts/data_pipeline.py", ROOT / "logs/training_metrics.csv",
    ROOT / "logs/training_config.json", ROOT / "logs/evaluation_ppl.json",
    ROOT / "curves/training_loss.png", ROOT / "models/best_checkpoint.json",
]
missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
train = json.loads((ROOT / "logs/training_config.json").read_text(encoding="utf-8"))
if config["amp"] or train["amp_enabled"]:
    raise SystemExit("part1 must keep AMP disabled")
if missing:
    raise SystemExit("missing: " + ", ".join(missing))
print("part1 reproducibility checks passed")
