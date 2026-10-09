"""Hand-written Unicode BPE with UTF-8 byte fallback for unseen characters.

Count adjacent pairs, repeatedly merge the highest-frequency pair, and retain
the initial alphabet.  Reserve 256 byte tokens for lossless unseen Unicode.
Only Python's standard library is required.

原始文本 → [Tokenizer 按规则切分] → token 序列 → [Vocab 查表] → ID 序列
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
BYTE_TOKENS = tuple(f"<0x{byte:02X}>" for byte in range(256))


class BPETokenizer:
    def __init__(self, vocab_size: int = 512, special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS) -> None:
        if not set(DEFAULT_SPECIAL_TOKENS).issubset(special_tokens):
            raise ValueError("special_tokens must include pad, unk, bos, and eos")
        if len(set(special_tokens)) != len(special_tokens) or set(special_tokens) & set(BYTE_TOKENS):
            raise ValueError("special tokens must be unique and distinct from byte tokens")
        if vocab_size < len(special_tokens) + 256:
            raise ValueError("vocab_size must accommodate special tokens and 256 fallback bytes")
        self.target_vocab_size = vocab_size
        self.special_tokens = tuple(special_tokens)
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []
        self.merge_history: list[dict] = []
        self.alphabet: set[str] = set()
        self._trained = False
        self._reset()

    def _reset(self) -> None:
        self.token_to_id = {}
        self.id_to_token = {}
        for token in (*self.special_tokens, *BYTE_TOKENS):
            self._add_token(token)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    def _add_token(self, token: str) -> None:
        if token not in self.token_to_id:
            index = len(self.token_to_id)
            self.token_to_id[token] = index
            self.id_to_token[index] = token

    @staticmethod
    def _pair_counts(sequences: Iterable[list[str]]) -> Counter[tuple[str, str]]:
        counts: Counter[tuple[str, str]] = Counter()
        for sequence in sequences:
            counts.update(zip(sequence, sequence[1:]))
        return counts

    @staticmethod
    def _merge_pair(sequence: list[str], pair: tuple[str, str]) -> list[str]:
        output: list[str] = []
        i = 0
        while i < len(sequence):
            if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
                output.append(pair[0] + pair[1])
                i += 2
            else:
                output.append(sequence[i])
                i += 1
        return output

    def train(self, texts: Iterable[str], *, vocab_size: int | None = None,
              min_pair_frequency: int = 2) -> "BPETokenizer":
        corpus = [list(text) for text in texts if text]
        if not corpus:
            raise ValueError("texts must contain non-empty strings")
        if min_pair_frequency < 1:
            raise ValueError("min_pair_frequency must be positive")
        target = self.target_vocab_size if vocab_size is None else vocab_size
        alphabet = {char for sequence in corpus for char in sequence}
        minimum = len(self.special_tokens) + 256 + len(alphabet)
        if target < minimum:
            raise ValueError(f"vocab_size must be at least {minimum} for this corpus and byte fallback")
        self._trained = False
        self.target_vocab_size = target
        self.alphabet = alphabet
        self.merges = []
        self.merge_history = []
        self._reset()
        for char in sorted(alphabet):
            self._add_token(char)
        reserved = set(self.special_tokens) | set(BYTE_TOKENS)
        while self.vocab_size < target:
            counts = self._pair_counts(corpus)
            candidates = [(pair, freq) for pair, freq in counts.items()
                          if pair[0] + pair[1] not in reserved]
            if not candidates:
                break
            pair, frequency = max(candidates, key=lambda item: (item[1], item[0]))
            if frequency < min_pair_frequency:
                break
            self.merges.append(pair)
            self._add_token(pair[0] + pair[1])
            self.merge_history.append({"step": len(self.merges), "left": pair[0],
                                       "right": pair[1], "pair_frequency": frequency,
                                       "merged_token": pair[0] + pair[1],
                                       "vocab_size": self.vocab_size})
            corpus = [self._merge_pair(sequence, pair) for sequence in corpus]
        self._trained = True
        return self

    def tokenize(self, text: str) -> list[str]:
        if not self._trained:
            raise RuntimeError("train or load the tokenizer first")
        tokens = list(text)
        for pair in self.merges:
            tokens = self._merge_pair(tokens, pair)
        output = []
        for token in tokens:
            if token in self.token_to_id:
                output.append(token)
            else:
                output.extend(BYTE_TOKENS[byte] for byte in token.encode("utf-8"))
        return output

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.token_to_id[token] for token in self.tokenize(text)]
        if add_bos:
            ids.insert(0, self.token_to_id["<bos>"])
        if add_eos:
            ids.append(self.token_to_id["<eos>"])
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        output = bytearray()
        byte_values = {token: byte for byte, token in enumerate(BYTE_TOKENS)}
        for index in ids:
            if index not in self.id_to_token:
                raise ValueError(f"unknown token id: {index}")
            token = self.id_to_token[index]
            if token in byte_values:
                output.append(byte_values[token])
            elif token in self.special_tokens and skip_special_tokens:
                if token == "<unk>":
                    raise ValueError("<unk> has no original text; encode uses byte fallback instead")
            else:
                output.extend(token.encode("utf-8"))
        return output.decode("utf-8")

    def save(self, path: str | Path) -> None:
        if not self._trained:
            raise RuntimeError("train before saving")
        payload = {"format_version": 2, "target_vocab_size": self.target_vocab_size,
                   "special_tokens": list(self.special_tokens), "alphabet": sorted(self.alphabet),
                   "vocab": self.id_to_token, "merges": self.merges,
                   "merge_history": self.merge_history}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 2:
            raise ValueError("legacy vocab is incompatible with byte fallback; retrain it")
        obj = cls(payload["target_vocab_size"], payload["special_tokens"])
        obj.id_to_token = {int(index): token for index, token in payload["vocab"].items()}
        obj.token_to_id = {token: index for index, token in obj.id_to_token.items()}
        obj.merges = [tuple(pair) for pair in payload["merges"]]
        obj.merge_history = payload["merge_history"]
        obj.alphabet = set(payload["alphabet"])
        obj._trained = True
        return obj


def train_tokenizer(texts: Iterable[str], vocab_size: int = 512) -> BPETokenizer:
    return BPETokenizer(vocab_size=vocab_size).train(texts)
