"""Evaluate a stage-three checkpoint with PPL and next-token accuracy."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_gpt" / "scripts"))
from minigpt import GPT, GPTConfig  # noqa: E402
from data_pipeline import build_token_stream, load_or_train_tokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "03_pretraining/data/raw/eng_fra_songs.txt")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "01_tokenizer/models/bpe_512.json")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = GPTConfig(**payload["model_config"]); model = GPT(model_config).to(device); model.load_state_dict(payload["model"]); model.eval()
    tokenizer = load_or_train_tokenizer(args.corpus, args.tokenizer)
    ids = torch.tensor(build_token_stream(args.corpus, tokenizer), dtype=torch.long)
    split = min(max(model_config.block_size + 2, int(len(ids) * .9)), len(ids) - model_config.block_size - 2); data = ids[split:]
    losses = []; correct = 0; total = 0
    with torch.no_grad():
        for _ in range(args.iters):
            starts = torch.randint(0, data.numel() - model_config.block_size - 1, (args.batch_size,))
            x = torch.stack([data[s:s + model_config.block_size] for s in starts]).to(device); y = torch.stack([data[s + 1:s + model_config.block_size + 1] for s in starts]).to(device)
            logits, loss = model(x, y); losses.append(float(loss)); correct += int((logits.argmax(-1) == y).sum()); total += y.numel()
    result = {"checkpoint": str(args.checkpoint), "validation_loss": sum(losses) / len(losses), "validation_ppl": math.exp(min(sum(losses) / len(losses), 20)), "next_token_accuracy": correct / total, "task": "held-out next-token proxy"}
    print(json.dumps(result, ensure_ascii=False, indent=2)); (ROOT / "03_pretraining/logs/evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
