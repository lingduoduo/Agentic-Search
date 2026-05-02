"""Simple vocabulary utilities for sequence tokenization."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20


class Vocabulary:
    """Track token-to-index mappings and token counts."""

    def __init__(self) -> None:
        self.token2idx: dict[str, int] = {}
        self.token2cnt: dict[str, int] = {}
        self.idx2token: dict[int, str] = {
            SOS_token: "SOS",
            EOS_token: "EOS",
        }
        self.num_token = 2

    def add_tokens(self, seq: str) -> None:
        """Split a sequence on whitespace and add each token."""

        for token in seq.split():
            self.create_vocab(token)

    def create_vocab(self, token: str) -> None:
        """Insert a token if missing, otherwise update its count."""

        if token not in self.token2idx:
            self.token2idx[token] = self.num_token
            self.token2cnt[token] = 1
            self.idx2token[self.num_token] = token
            self.num_token += 1
            return

        self.token2cnt[token] += 1


def normalize_text(text: str, *, keep_non_ascii: bool = False) -> str:
    """Normalize a sentence for lightweight token-based indexing."""

    normalized = text.lower()
    if not keep_non_ascii:
        normalized = unicodedata.normalize("NFD", normalized)
        normalized = normalized.encode("ascii", errors="ignore").decode("utf-8")
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize_text(
    text: str,
    *,
    keep_non_ascii: bool = False,
    max_length: int | None = None,
) -> list[str]:
    """Normalize text and split it into tokens."""

    tokens = normalize_text(text, keep_non_ascii=keep_non_ascii).split()
    if max_length is not None:
        return tokens[:max_length]
    return tokens


def build_vocabulary_from_sequences(
    sequences: Iterable[str],
    *,
    keep_non_ascii: bool = False,
    max_length: int = MAX_LENGTH,
) -> Vocabulary:
    """Create a vocabulary from an iterable of text sequences."""

    vocabulary = Vocabulary()
    for sequence in sequences:
        tokens = tokenize_text(
            sequence,
            keep_non_ascii=keep_non_ascii,
            max_length=max_length,
        )
        if tokens:
            vocabulary.add_tokens(" ".join(tokens))
    return vocabulary


def extract_keywords(
    text: str,
    *,
    limit: int = 10,
    keep_non_ascii: bool = False,
    max_length: int = MAX_LENGTH,
) -> list[str]:
    """Return the most frequent normalized tokens from a document."""

    if limit < 1:
        return []

    tokens = tokenize_text(
        text,
        keep_non_ascii=keep_non_ascii,
        max_length=max_length,
    )
    counts = Counter(tokens)
    return [token for token, _ in counts.most_common(limit)]
