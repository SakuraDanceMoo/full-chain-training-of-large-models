"""Train and inspect the stage-one tokenizer on the real song corpus."""

from pathlib import Path

from corpus import default_song_corpus_path, load_parallel_texts
from tokenizer import BPETokenizer


def main() -> None:
    corpus_path = default_song_corpus_path()
    corpus = load_parallel_texts(corpus_path)
    tokenizer = BPETokenizer(vocab_size=512).train(corpus, min_pair_frequency=1)
    sample = corpus[0]
    ids = tokenizer.encode(sample, add_bos=True, add_eos=True)

    print(f"corpus: {corpus_path}")
    print(f"sentences: {len(corpus)}")
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"merge rules: {len(tokenizer.merges)}")
    print(f"tokens: {tokenizer.tokenize(sample)}")
    print(f"ids: {ids}")
    print(f"decoded: {tokenizer.decode(ids)}")
    print(f"lossless: {tokenizer.decode(tokenizer.encode(sample)) == sample}")

    output = Path(__file__).resolve().parents[1] / "models" / "bpe_full_corpus_512.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output)
    print(f"saved vocabulary: {output}")


if __name__ == "__main__":
    main()
