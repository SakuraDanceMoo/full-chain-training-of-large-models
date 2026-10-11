"""Reproducible small-scale pretraining loop for the 0.1B-ready Mini-GPT.

The default ``--profile smoke`` is intentionally small and runs on a single
CPU/GPU. ``--profile 0.1b`` selects a roughly 100M-parameter configuration.
The loop demonstrates gradient accumulation, AMP and activation checkpointing
so the same project can be scaled after the smoke run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
GPT_DIR = ROOT / "02_gpt" / "scripts"
if str(GPT_DIR) not in sys.path:
    sys.path.insert(0, str(GPT_DIR))
from minigpt import GPT, GPTConfig  # noqa: E402
from data_pipeline import build_token_stream, load_or_train_tokenizer  # noqa: E402


@dataclass
class TrainConfig:
    profile: str
    block_size: int
    layers: int
    heads: int
    embed_dim: int
    batch_size: int
    grad_accum: int
    steps: int
    eval_interval: int
    eval_iters: int
    learning_rate: float
    weight_decay: float
    amp: bool
    activation_checkpointing: bool
    seed: int


PROFILES = {
    "smoke": dict(block_size=64, layers=2, heads=4, embed_dim=128, batch_size=4, grad_accum=2, steps=80),
    "0.1b": dict(block_size=512, layers=12, heads=12, embed_dim=768, batch_size=2, grad_accum=16, steps=10000),
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--corpus", type=Path, default=ROOT / "03_pretraining/data/raw/eng_fra_songs.txt")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "01_tokenizer/models/bpe_512.json")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-iters", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", type=Path, default=ROOT / "03_pretraining")
    return parser.parse_args()


def profile_config(args: argparse.Namespace) -> TrainConfig:
    values = dict(PROFILES[args.profile])
    if args.steps is not None:
        values["steps"] = args.steps
    return TrainConfig(profile=args.profile, **values, eval_interval=args.eval_interval,
                       eval_iters=args.eval_iters, learning_rate=args.learning_rate,
                       weight_decay=args.weight_decay, amp=args.amp,
                       activation_checkpointing=args.activation_checkpointing, seed=args.seed)


def sample_batch(data: Tensor, block_size: int, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    if data.numel() <= block_size + 1:
        raise ValueError("token stream is too short for the selected block_size")
    starts = torch.randint(0, data.numel() - block_size - 1, (batch_size,))
    x = torch.stack([data[start:start + block_size] for start in starts]).to(device)
    y = torch.stack([data[start + 1:start + block_size + 1] for start in starts]).to(device)
    return x, y


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def evaluate(model: GPT, data: Tensor, config: TrainConfig, model_config: GPTConfig,
             device: torch.device, iters: int) -> tuple[float, float]:
    model.eval(); losses = []; correct = 0; total = 0
    for _ in range(iters):
        x, y = sample_batch(data, model_config.block_size, config.batch_size, device)
        logits, loss = model(x, y)
        losses.append(float(loss))
        correct += int((logits.argmax(dim=-1) == y).sum())
        total += y.numel()
    model.train()
    mean_loss = sum(losses) / len(losses)
    return mean_loss, correct / total


def main() -> None:
    args = get_args(); config = profile_config(args)
    random.seed(config.seed); torch.manual_seed(config.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(config.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = load_or_train_tokenizer(args.corpus, args.tokenizer)
    ids = torch.tensor(build_token_stream(args.corpus, tokenizer, args.output / "data/processed/token_ids.json"), dtype=torch.long)
    split = min(max(config.block_size + 2, int(len(ids) * 0.9)), len(ids) - config.block_size - 2)
    train_data, val_data = ids[:split], ids[split:]
    model_config = GPTConfig(tokenizer.vocab_size, config.block_size, config.layers, config.heads, config.embed_dim, 0.1)
    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    metrics = []
    output_logs, output_curves, output_models = args.output / "logs", args.output / "curves", args.output / "models"
    output_logs.mkdir(parents=True, exist_ok=True); output_curves.mkdir(parents=True, exist_ok=True); output_models.mkdir(parents=True, exist_ok=True)
    for step in range(config.steps + 1):
        if step % config.eval_interval == 0 or step == config.steps:
            train_loss, train_acc = evaluate(model, train_data, config, model_config, device, config.eval_iters)
            val_loss, val_acc = evaluate(model, val_data, config, model_config, device, config.eval_iters)
            row = {"step": step, "train_loss": train_loss, "val_loss": val_loss,
                   "train_ppl": math.exp(min(train_loss, 20)), "val_ppl": math.exp(min(val_loss, 20)),
                   "train_next_token_accuracy": train_acc, "val_next_token_accuracy": val_acc}
            metrics.append(row); print(json.dumps(row, ensure_ascii=False))
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                        "model_config": asdict(model_config), "train_config": asdict(config),
                        "tokenizer_path": str(args.tokenizer)}, output_models / f"checkpoint_step_{step:06d}.pt")
        if step == config.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.grad_accum):
            x, y = sample_batch(train_data, model_config.block_size, config.batch_size, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y, activation_checkpointing=config.activation_checkpointing)
                loss = loss / config.grad_accum
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
    with (output_logs / "training_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    training_record = {"device": str(device), "amp_enabled": use_amp, "activation_checkpointing": config.activation_checkpointing,
                       "parameters": count_parameters(model), "tokenizer_vocab_size": tokenizer.vocab_size,
                       "token_count": len(ids), "train_tokens": len(train_data), "validation_tokens": len(val_data),
                       "model_config": asdict(model_config), "train_config": asdict(config)}
    (output_logs / "training_config.json").write_text(json.dumps(training_record, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_logs / "ppl_and_downstream.json").write_text(json.dumps({"final": metrics[-1], "task": "held-out next-token proxy", "note": "This is a lightweight downstream proxy, not a benchmark claim."}, ensure_ascii=False, indent=2), encoding="utf-8")
    figure, axis = plt.subplots(figsize=(7, 4)); axis.plot([m["step"] for m in metrics], [m["train_loss"] for m in metrics], marker="o", label="train"); axis.plot([m["step"] for m in metrics], [m["val_loss"] for m in metrics], marker="o", label="validation"); axis.set(xlabel="optimization step", ylabel="cross-entropy loss", title=f"Stage 3 pretraining ({config.profile})"); axis.grid(alpha=.25); axis.legend(); figure.tight_layout(); figure.savefig(output_curves / "training_loss.png", dpi=160); plt.close(figure)


if __name__ == "__main__":
    main()
