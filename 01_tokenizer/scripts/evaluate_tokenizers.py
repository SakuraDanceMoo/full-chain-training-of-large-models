"""Reproducible song-corpus BPE/official SentencePiece Unigram comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from corpus import default_song_corpus_path, load_parallel_texts
from sentencepiece_tokenizer import SentencePieceTokenizer
from tokenizer import BPETokenizer

DEFAULT_VOCAB_SIZES = (384, 512, 640, 768, 1024)


def split_pairs(texts: list[str], seed: int = 42) -> tuple[list[str], list[str]]:
    """Keep aligned pairs and repeated lyrics in the same split.

    Group pairs connected by any identical sentence before shuffling. Songs
    repeat chorus lines; a plain random row split leaks those exact sentences.
    """
    pairs = list(zip(texts[::2], texts[1::2]))
    parent = list(range(len(pairs)))
    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    first_seen = {}
    for index, pair in enumerate(pairs):
        for sentence in pair:
            if sentence in first_seen:
                parent[find(index)] = find(first_seen[sentence])
            else:
                first_seen[sentence] = index
    groups = defaultdict(list)
    for index, pair in enumerate(pairs):
        groups[find(index)].extend(pair)
    shuffled = list(groups.values())
    random.Random(seed).shuffle(shuffled)
    target = int(len(texts) * 0.8)
    training, held_out = [], []
    for group in shuffled:
        (training if len(training) + len(group) <= target else held_out).extend(group)
    assert set(training).isdisjoint(held_out)
    return training, held_out


def measure(tokenizer, texts: list[str]) -> dict:
    total_chars = total_tokens = mismatches = unknown_tokens = 0
    for text in texts:
        ids = tokenizer.encode(text)
        total_chars += len(text)
        total_tokens += len(ids)
        unknown_tokens += ids.count(1)
        mismatches += tokenizer.decode(ids) != text
    return {"sentences": len(texts), "characters": total_chars,
            "tokens": total_tokens, "avg_chars_per_token": total_chars / total_tokens,
            "round_trip_failures": mismatches, "unknown_tokens": unknown_tokens}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=default_song_corpus_path())
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=DEFAULT_VOCAB_SIZES)
    args = parser.parse_args()
    texts = load_parallel_texts(args.corpus)
    training, held_out = split_pairs(texts)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    models, logs, curves = (out / name for name in ("models", "logs", "curves"))
    for directory in (models, logs, curves):
        directory.mkdir(parents=True, exist_ok=True)
    corpus_hash = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    pair_counts = BPETokenizer._pair_counts([list(s) for s in training])
    write_csv(logs / "initial_pair_frequencies.csv", [
        {"left": pair[0], "right": pair[1], "frequency": freq}
        for pair, freq in sorted(pair_counts.items(), key=lambda x: (-x[1], x[0]))
    ])
    rows = []
    print(f"{len(texts)//2} pairs, {len(training)} training / {len(held_out)} held-out sentences", flush=True)
    for size in sorted(set(args.vocab_sizes)):
        for name in ("bpe", "sentencepiece_unigram"):
            if name == "bpe":
                tokenizer = BPETokenizer(size).train(training, min_pair_frequency=1)
                tokenizer.save(models / f"bpe_{size}.json")
                write_csv(logs / f"bpe_{size}_merge_history.csv", tokenizer.merge_history)
            else:
                tokenizer = SentencePieceTokenizer.train(training, size, models / f"unigram_{size}")
            for split, corpus in (("train", training), ("held_out", held_out), ("all", texts)):
                row = {"model": name, "split": split, "requested_vocab_size": size,
                       "actual_vocab_size": tokenizer.vocab_size, **measure(tokenizer, corpus)}
                if row["round_trip_failures"] or row["unknown_tokens"]:
                    raise AssertionError(f"lossless/unknown-token check failed: {row}")
                rows.append(row)
            print(f"{name}: requested={size}, actual={tokenizer.vocab_size}, "
                  f"held_out chars/token={rows[-2]['avg_chars_per_token']:.4f}", flush=True)
    write_csv(logs / "tokenizer_compression_results.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180, sharey=True)
    for ax, split, title in zip(axes, ("train", "held_out"), ("Training split", "Held-out split")):
        for name, color, label in (("bpe", "#2C6EAA", "Hand-written Unicode BPE"),
                                   ("sentencepiece_unigram", "#A24F70", "Official SentencePiece Unigram")):
            points = [row for row in rows if row["model"] == name and row["split"] == split]
            ax.plot([p["actual_vocab_size"] for p in points], [p["avg_chars_per_token"] for p in points],
                    marker="o", label=label, color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Actual vocabulary size (includes specials + 256 byte tokens)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Characters per token (higher means fewer tokens)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("eng_fra_songs.txt: vocabulary size vs compression", fontsize=14)
    fig.tight_layout()
    fig.savefig(curves / "tokenizer_compression_curve.png")
    fig.savefig(curves / "tokenizer_compression_curve.svg")
    plt.close(fig)
    example_models = {"bpe": BPETokenizer.load(models / f"bpe_{max(args.vocab_sizes)}.json"),
                      "sentencepiece_unigram": SentencePieceTokenizer.load(models / f"unigram_{max(args.vocab_sizes)}.model")}
    examples = [{"text": text, "tokenizers": {name: {"pieces": model.tokenize(text),
                "ids": model.encode(text), "decoded": model.decode(model.encode(text))}
                for name, model in example_models.items()}} for text in held_out[:6]]
    (logs / "tokenization_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        corpus_label = args.corpus.resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix()
    except ValueError:
        corpus_label = args.corpus.name
    metadata = {"corpus": corpus_label, "sha256": corpus_hash,
                "pairs": len(texts)//2, "seed": 42, "training_sentences": len(training),
                "split_method": "group aligned pairs connected by exact repeated sentences, then 80/20",
                "exact_sentence_overlap": len(set(training) & set(held_out)),
                "held_out_sentences": len(held_out), "vocab_sizes": args.vocab_sizes,
                "metric": "sum Unicode code points / sum encoded tokens, without BOS/EOS",
                "sentencepiece_version": __import__("sentencepiece").__version__,
                "raw_corpus_unchanged": hashlib.sha256(args.corpus.read_bytes()).hexdigest() == corpus_hash}
    (logs / "experiment_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Tokenizer comparison results", "", "Corpus: `eng_fra_songs.txt`.",
             f"{len(texts)//2} aligned pairs; seed 42; {len(training)} training and {len(held_out)} held-out sentences.",
             "Both languages of each pair remain in the same split. No model is trained on held-out sentences.",
             "Pairs connected by identical repeated lyrics are grouped before splitting; exact sentence overlap is zero.",
             "", "Metric: total Unicode characters / total tokens, excluding inserted BOS/EOS. Higher means more compression.",
             "Vocabulary size includes four special tokens and 256 reserved byte tokens in both models.",
             "", "| Model | Requested vocabulary | Actual vocabulary | Train chars/token | Held-out chars/token |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        if row["split"] == "train":
            test = next(p for p in rows if p["model"] == row["model"] and p["requested_vocab_size"] == row["requested_vocab_size"] and p["split"] == "held_out")
            lines.append(f"| {row['model']} | {row['requested_vocab_size']} | {row['actual_vocab_size']} | {row['avg_chars_per_token']:.4f} | {test['avg_chars_per_token']:.4f} |")
    lines.extend(["", "All measured corpus round trips succeeded; no unknown tokens were emitted.",
                  "These small-corpus results measure compression, not downstream model quality.",
                  "The algorithms can produce different segmentations; matching token IDs is not expected.",
                  "Unigram selects a probable segmentation from a trained probabilistic vocabulary; BPE follows merge ranks."])
    (logs / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved models, frequency logs, curves, and report to {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
