"""Corpus loading helpers for the tokenizer experiments."""

from __future__ import annotations

from pathlib import Path


def load_parallel_texts(path: str | Path) -> list[str]:
    """Load both sides of a tab-separated English/French corpus.

    The original song corpus stores one English sentence and one French
    sentence per line.  Keeping both columns makes the tokenizer vocabulary
    representative of the data used by the encoder-decoder examples.
    """

    texts: list[str] = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        fields = raw_line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"expected two tab-separated fields at line {line_number}")
        # Preserve each sentence verbatim; trim neither spaces nor accents.
        texts.extend(fields)
    if not texts:
        raise ValueError(f"no text found in corpus: {path}")
    return texts


def default_song_corpus_path() -> Path:
    """Return the repository's real stage-one training corpus path."""

    return (
        Path(__file__).resolve().parent.parent
        / "data"
        / "raw"
        / "eng_fra_songs.txt"
    )
