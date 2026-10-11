"""Stage-three data pipeline using the stage-one reversible BPE tokenizer."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Iterator


PART_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
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
        if child.is_file() and child.name not in {"manifest.json", "README.md"} and child.suffix.lower() in allowed:
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
