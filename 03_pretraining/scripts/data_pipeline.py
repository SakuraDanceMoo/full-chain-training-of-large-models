"""Stage-three data pipeline using the stage-one reversible BPE tokenizer."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_DIR = ROOT / "01_tokenizer" / "scripts"
if str(TOKENIZER_DIR) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_DIR))
from tokenizer import BPETokenizer  # noqa: E402


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
    tokenizer = BPETokenizer(vocab_size=vocab_size).train(load_parallel_texts(corpus))
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tokenizer_path)
    return tokenizer


def build_token_stream(corpus: Path, tokenizer: BPETokenizer, cache_path: Path | None = None) -> list[int]:
    if cache_path and cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("corpus_sha256") == hashlib.sha256(corpus.read_bytes()).hexdigest():
            return [int(value) for value in payload["ids"]]
    ids: list[int] = []
    for text in load_parallel_texts(corpus):
        ids.extend(tokenizer.encode(text, add_eos=True))
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "ids": ids,
        }), encoding="utf-8")
    return ids
