"""Official SentencePiece Unigram baseline with lossless whitespace options.

SentencePiece supports Unigram and BPE, not WordPiece.  This baseline uses
the official library's probabilistic training and Viterbi segmentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import sentencepiece as spm


class SentencePieceTokenizer:
    def __init__(self, processor: spm.SentencePieceProcessor) -> None:
        self.processor = processor

    @property
    def vocab_size(self) -> int:
        return self.processor.get_piece_size()

    def tokenize(self, text: str) -> list[str]:
        return [self.processor.id_to_piece(index) for index in self.encode(text)]

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        # SentencePiece reserves U+2581 as its space marker. A literal marker
        # would otherwise become a space on decode. Encode such input as raw
        # fallback bytes, which the official decoder reconstructs verbatim.
        if "\u2581" in text:
            ids = [self.processor.piece_to_id(f"<0x{byte:02X}>") for byte in text.encode("utf-8")]
            if add_bos:
                ids.insert(0, self.processor.bos_id())
            if add_eos:
                ids.append(self.processor.eos_id())
            return ids
        return self.processor.encode(text, out_type=int, add_bos=add_bos, add_eos=add_eos)

    def decode(self, ids: Iterable[int]) -> str:
        return self.processor.decode(list(ids))

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, model_prefix: str | Path) -> "SentencePieceTokenizer":
        prefix = Path(model_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(texts), model_prefix=str(prefix), model_type="unigram",
            vocab_size=vocab_size, character_coverage=1.0, byte_fallback=True,
            normalization_rule_name="identity", add_dummy_prefix=False,
            remove_extra_whitespaces=False, split_by_whitespace=False,
            allow_whitespace_only_pieces=True,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            hard_vocab_limit=False, shuffle_input_sentence=False, num_threads=1,
            minloglevel=2,
        )
        return cls.load(prefix.with_suffix(".model"))

    @classmethod
    def load(cls, path: str | Path) -> "SentencePieceTokenizer":
        return cls(spm.SentencePieceProcessor(model_file=str(path)))
