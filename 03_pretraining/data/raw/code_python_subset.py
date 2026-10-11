# Mini-GPT source subset
"""A small decoder-only GPT trained on tinyShakespeare.

The implementation is intentionally self-contained: it uses PyTorch only and
keeps the same attention -> FFN -> residual/LayerNorm decomposition as the
encoder block in ``09_encoder_decoder/scripts/transformer.py``.  Unlike that
bidirectional encoder, this block applies a strict causal upper-triangular mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "02_gpt" / "data" / "raw" / "tinyshakespeare.txt"


class CharTokenizer:
    """Deterministic character vocabulary; every input character is reversible."""

    def __init__(self, text: str) -> None:
        self.itos = sorted(set(text))
        if not self.itos:
            raise ValueError("training text is empty")
        self.stoi = {char: index for index, char in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            raise ValueError(f"text contains characters absent from vocabulary: {unknown!r}")
        return [self.stoi[char] for char in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(index)] for index in ids)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a fixed lower-triangular visibility mask."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)))

    def forward(self, x: Tensor) -> Tensor:
        batch, steps, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=-1)
        shape = (batch, steps, self.n_head, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:steps, :steps], torch.finfo(scores.dtype).min)
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        output = (weights @ v).transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.resid_dropout(self.proj(output))


class GPTBlock(nn.Module):
    """Causal self-attention + FFN + residual LayerNorm, based on EncoderBlock."""

    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config.n_embd, config.n_head, config.block_size, config.dropout)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class GPT(nn.Module):
    """Embedding + learned positions + N GPTBlocks + output projection."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: Tensor, targets: Tensor | None = None,
                target_mask: Tensor | None = None,
                activation_checkpointing: bool = False) -> tuple[Tensor, Tensor | None]:
        batch, steps = idx.shape
        if steps > self.config.block_size:
            raise ValueError(f"sequence length {steps} exceeds block_size {self.config.block_size}")
        positions = torch.arange(steps, device=idx.device)
        x = self.dropout(self.token_embedding(idx) + self.position_embedding(positions))
        for block in self.blocks:
            if activation_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = next_token_loss(logits, targets, target_mask) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = 20, do_sample: bool = True) -> Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probabilities = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probabilities, 1) if do_sample else probabilities.argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, next_id), dim=1)
        if was_training:
            self.train()
        return idx


def next_token_loss(logits: Tensor, targets: Tensor | None, target_mask: Tensor | None = None) -> Tensor:
    """Cross entropy for shifted next-token labels, optionally ignoring padding."""
    if targets is None:
        raise ValueError("targets are required for next-token loss")
    if logits.shape[:2] != targets.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and targets {tuple(targets.shape)} disagree")
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    if target_mask is None:
        return F.cross_entropy(flat_logits, flat_targets)
    valid = target_mask.reshape(-1).bool()
    if valid.sum() == 0:
        raise ValueError("target_mask contains no valid tokens")
    return F.cross_entropy(flat_logits[valid], flat_targets[valid])


def batches(data: Tensor, block_size: int, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    if data.numel() <= block_size + 1:
        raise ValueError("dataset must contain more than block_size + 1 tokens")
    starts = torch.randint(0, data.numel() - block_size - 1, (batch_size,))
    x = torch.stack([data[start:start + block_size] for start in starts]).to(device)
    y = torch.stack([data[start + 1:start + block_size + 1] for start in starts]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model: GPT, splits: dict[str, Tensor], config: GPTConfig,
                  batch_size: int, eval_iters: int, device: torch.device) -> dict[str, float]:
    model.eval()
    result = {}
    for name, data in splits.items():
        losses = []
        for _ in range(eval_iters):
            x, y = batches(data, config.block_size, batch_size, device)
            _, loss = model(x, y)
            losses.append(float(loss))
        result[name] = sum(losses) / len(losses)
    model.train()
    return result


def train(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    text = args.data.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(0.9 * len(ids))
    splits = {"train": ids[:split], "val": ids[split:]}
    config = GPTConfig(tokenizer.vocab_size, args.block_size, args.layers, args.heads, args.embed_dim, args.dropout)
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    metrics: list[dict] = []
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            values = estimate_loss(model, splits, config, args.batch_size, args.eval_iters, device)
            row = {"step": step, **values}
            metrics.append(row)
            print(f"step {step:5d} | train {values['train']:.4f} | val {values['val']:.4f}")
        if step == args.steps:
            break
        x, y = batches(splits["train"], config.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    args.output_logs.mkdir(parents=True, exist_ok=True)
    args.output_curves.mkdir(parents=True, exist_ok=True)
    args.output_models.mkdir(parents=True, exist_ok=True)
    with (args.output_logs / "training_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "train", "val"])
        writer.writeheader(); writer.writerows(metrics)
    (args.output_logs / "training_config.json").write_text(json.dumps({**vars(args), "device": str(device), "config": asdict(config)}, default=str, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "config": asdict(config), "itos": tokenizer.itos}, args.output_models / "minigpt_tinyshakespeare.pt")
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([row["step"] for row in metrics], [row["train"] for row in metrics], marker="o", label="train")
    axis.plot([row["step"] for row in metrics], [row["val"] for row in metrics], marker="o", label="validation")
    axis.set(xlabel="optimization step", ylabel="cross-entropy loss", title="Mini-GPT on tinyShakespeare")
    axis.grid(alpha=0.25); axis.legend(); figure.tight_layout()
    figure.savefig(args.output_curves / "training_loss.png", dpi=160); plt.close(figure)
    prompt = "ROMEO:\n"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    sample_ids = model.generate(prompt_ids, args.sample_tokens, args.temperature, args.top_k)[0].tolist()
    sample = tokenizer.decode(sample_ids)
    (args.output_logs / "generation_samples.txt").write_text(f"prompt: {prompt!r}\n\n{sample}\n", encoding="utf-8")
    summary = {"device": str(device), "parameters": sum(p.numel() for p in model.parameters()), "vocab_size": tokenizer.vocab_size, "metrics": metrics, "sample": sample}
    (args.output_logs / "stage2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-logs", type=Path, default=ROOT / "02_gpt/logs")
    parser.add_argument("--output-curves", type=Path, default=ROOT / "02_gpt/curves")
    parser.add_argument("--output-models", type=Path, default=ROOT / "02_gpt/models")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

# Data pipeline subset
"""Stage-three data pipeline using the stage-one reversible BPE tokenizer."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_DIR = ROOT / "01_tokenizer" / "scripts"
if str(TOKENIZER_DIR) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_DIR))
from tokenizer import BPETokenizer  # noqa: E402


def iter_corpus_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(path)
    allowed = {".txt", ".py", ".md", ".json"}
    for child in sorted(path.rglob("*")):
        if child.is_file() and child.suffix.lower() in allowed:
            yield child


def corpus_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    files = list(iter_corpus_files(path))
    if not files:
        raise ValueError(f"no corpus files found in {path}")
    for file_path in files:
        name = file_path.relative_to(path).as_posix() if path.is_dir() else file_path.name
        digest.update(name.encode("utf-8")); digest.update(b"\0"); digest.update(file_path.read_bytes())
    return digest.hexdigest()


def load_corpus_texts(path: Path) -> list[str]:
    """Load a file or a deterministic directory of text/code documents."""
    texts = [file_path.read_text(encoding="utf-8") for file_path in iter_corpus_files(path)]
    if not texts:
        raise ValueError(f"no text found in {path}")
    return texts


def load_parallel_texts(path: Path) -> list[str]:
    texts: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split("\t")
        if len(fields) != 2:
            raise ValueError(f"line {line_number} must contain English and French tab-separated fields")
        texts.extend(fields)
    if not texts:
        raise ValueError(f"no samples found in {path}")
    return texts


def load_or_train_tokenizer(corpus: Path, tokenizer_path: Path, vocab_size: int = 512) -> BPETokenizer:
    if tokenizer_path.is_file():
        return BPETokenizer.load(tokenizer_path)
    tokenizer = BPETokenizer(vocab_size=vocab_size).train(load_corpus_texts(corpus))
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tokenizer_path)
    return tokenizer


def build_token_stream(corpus: Path, tokenizer: BPETokenizer, cache_path: Path | None = None) -> list[int]:
    if cache_path and cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("corpus_sha256") == corpus_fingerprint(corpus):
            return [int(value) for value in payload["ids"]]
    ids: list[int] = []
    for text in load_corpus_texts(corpus):
        ids.extend(tokenizer.encode(text, add_eos=True))
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "corpus_sha256": corpus_fingerprint(corpus),
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "ids": ids,
        }), encoding="utf-8")
    return ids
