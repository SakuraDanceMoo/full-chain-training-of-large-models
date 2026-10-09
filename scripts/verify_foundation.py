"""Import-check inherited foundation modules without starting model training."""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "09_encoder_decoder" / "scripts"))
for name in ("machine_translation", "encoder_decoder", "attention_mechanism", "transformer", "bert", "seq2seq"):
    importlib.import_module(name)
    print(f"Imported {name}")
assert (ROOT / "09_encoder_decoder/data/raw/eng_fra_songs.txt").is_file()
print("Foundation imports and corpus path passed; training not started.")
