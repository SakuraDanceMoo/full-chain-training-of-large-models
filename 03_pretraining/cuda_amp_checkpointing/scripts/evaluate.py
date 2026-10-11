"""Evaluate a stage-three checkpoint with PPL and next-token accuracy."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PART_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "02_gpt" / "scripts"))
from minigpt import GPT, GPTConfig  # noqa: E402
from data_pipeline import build_token_stream, load_or_train_tokenizer, load_parallel_texts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=PART_ROOT / "data/raw")
    parser.add_argument("--task", choices=("ppl", "translation"), default="ppl")
    parser.add_argument("--translation-data", type=Path, default=None,
                        help="Optional translation file; this corpus intentionally has no English-French songs.")
    parser.add_argument("--tokenizer", type=Path, default=PART_ROOT / "models/tokenizer_bpe_512.json")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.checkpoint.suffix == ".json":
        manifest = json.loads(args.checkpoint.read_text(encoding="utf-8")); payload = manifest
        state = {}
        for part in manifest["parts"]:
            state.update(torch.load(args.checkpoint.parent / part, map_location="cpu", weights_only=False)["model"])
    else:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False); state = payload["model"]
    model_config = GPTConfig(**payload["model_config"]); model = GPT(model_config).to(device); model.load_state_dict(state); model.eval()
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
    if args.task == "translation":
        if args.translation_data is None:
            raise ValueError("cuda_amp_checkpointing has no English-French translation corpus; use --task ppl")
        pairs = []
        for line in args.translation_data.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) == 2: pairs.append(fields)
        token_correct = token_total = exact = 0
        for source, target in pairs[:20]:
            prefix = tokenizer.encode(source + "\t")
            target_ids = tokenizer.encode(target, add_eos=True)
            sequence = (prefix + target_ids)[-model_config.block_size:]
            prefix_len = min(len(prefix), len(sequence)); x = torch.tensor([sequence], device=device)
            logits, _ = model(x); predictions = logits[0, prefix_len - 1:-1].argmax(-1).tolist()[:len(target_ids)]
            token_correct += sum(a == b for a, b in zip(predictions, target_ids)); token_total += len(target_ids)
            exact += int(predictions == target_ids)
        result.update({"task": "English-to-French teacher-forced translation mini-task", "translation_token_accuracy": token_correct / max(token_total, 1), "translation_exact_match": exact / max(min(len(pairs), 20), 1), "translation_examples": min(len(pairs), 20)})
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    output = args.output or (ROOT / "03_pretraining/logs" / f"evaluation_{args.task}.json")
    output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
