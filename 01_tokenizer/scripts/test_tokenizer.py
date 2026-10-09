"""Verify round trips, pair merges, Unicode fallback and model persistence."""

from __future__ import annotations

import tempfile
from pathlib import Path

from corpus import default_song_corpus_path, load_parallel_texts
from evaluate_tokenizers import split_pairs
from sentencepiece_tokenizer import SentencePieceTokenizer
from tokenizer import BPETokenizer


def main() -> None:
    texts = load_parallel_texts(default_song_corpus_path())
    train, test = split_pairs(texts)
    assert len(texts) == 1132 and len(train) + len(test) == len(texts)
    assert set(train).isdisjoint(test), "repeated song lyrics must not leak into held-out data"
    assert BPETokenizer._pair_counts([list("aaaa")])[("a", "a")] == 3
    assert BPETokenizer._merge_pair(list("aaaaa"), ("a", "a")) == ["aa", "aa", "a"]
    small = BPETokenizer(266).train(["ababab"], min_pair_frequency=1)
    assert small.merge_history[0]["left"] == "a" and small.merge_history[0]["right"] == "b"
    assert small.merge_history[0]["pair_frequency"] == 3
    probes = ["", "  hello  world  ", "\tfirst\r\nsecond\t", "中文🙂🧪", "é e\u0301", "<bos><unk><0x00>", "literal ▁ boundary"]
    with tempfile.TemporaryDirectory(prefix="tokenizer_checks_") as directory:
        out = Path(directory)
        bpe = BPETokenizer(384).train(train, min_pair_frequency=1)
        unigram = SentencePieceTokenizer.train(train, 384, out / "unigram")
        for name, tokenizer in (("bpe", bpe), ("unigram", unigram)):
            for text in texts + probes:
                ids = tokenizer.encode(text)
                decoded = tokenizer.decode(ids)
                assert decoded == text, (name, repr(text), repr(decoded))
                assert 1 not in ids, (name, "unknown token emitted")
                assert tokenizer.decode(tokenizer.encode(text, add_bos=True, add_eos=True)) == text
            print(f"{name}: {len(texts)} corpus sentences + {len(probes)} edge cases passed", flush=True)
        bpe.save(out / "bpe.json")
        restored_models = (BPETokenizer.load(out / "bpe.json"), SentencePieceTokenizer.load(out / "unigram.model"))
        for model, restored in zip((bpe, unigram), restored_models):
            for text in test + probes:
                assert restored.encode(text) == model.encode(text)
                assert restored.decode(restored.encode(text)) == text
        try:
            BPETokenizer(260).train(texts)
        except ValueError:
            pass
        else:
            raise AssertionError("an undersized vocabulary must be rejected")
    print("All round-trip, persistence, byte fallback and BPE merge checks passed", flush=True)


if __name__ == "__main__":
    main()
