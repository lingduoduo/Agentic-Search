"""Dependency-free WordPiece tokenization over a BERT-style vocabulary.

This module deliberately imports neither torch nor transformers. The intent
model reads pretrained wordpieces at serving time, and keeping that path free of
heavy dependencies is what lets the unit-test CI job — which installs neither —
actually exercise it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from src.internal.document_index.text import normalize_text

PAD_ID = 0
UNK_ID = 100
# HuggingFace's `max_input_chars_per_word`: a longer run of characters is
# treated as unknown rather than scanned.
MAX_CHARS_PER_WORD = 100

_PAD_TOKEN = "[PAD]"
_UNK_TOKEN = "[UNK]"
_CONTINUATION = "##"


class WordPieceVocabulary:
    """Greedy longest-match-first WordPiece over an id-ordered vocabulary."""

    def __init__(self, token_to_id: dict[str, int]) -> None:
        self._token_to_id = token_to_id

    @classmethod
    def from_tokens(cls, tokens: Sequence[str]) -> "WordPieceVocabulary":
        """Build from tokens whose position in the sequence is their id."""
        token_to_id = {token: index for index, token in enumerate(tokens)}
        for token, expected in ((_PAD_TOKEN, PAD_ID), (_UNK_TOKEN, UNK_ID)):
            if token_to_id.get(token) != expected:
                raise ValueError(
                    f"Vocabulary must place {token} at id {expected}; found "
                    f"{token_to_id.get(token)}"
                )
        return cls(token_to_id)

    @classmethod
    def from_file(cls, path: Path) -> "WordPieceVocabulary":
        """Read a vocab.txt of one token per line, ordered by id."""
        tokens = path.read_text(encoding="utf-8").split("\n")
        while tokens and not tokens[-1]:
            tokens.pop()
        if not tokens:
            raise ValueError(f"WordPiece vocabulary file is empty: {path}")
        return cls.from_tokens(tokens)

    @property
    def size(self) -> int:
        return len(self._token_to_id)

    def encode(self, text: str) -> list[int]:
        """Normalize, split on whitespace, and encode each word."""
        ids: list[int] = []
        for word in normalize_text(text).split():
            ids.extend(self._encode_word(word))
        return ids

    def _encode_word(self, word: str) -> list[int]:
        if len(word) > MAX_CHARS_PER_WORD:
            return [UNK_ID]

        pieces: list[int] = []
        start = 0
        while start < len(word):
            end = len(word)
            match: int | None = None
            while start < end:
                candidate = word[start:end]
                if start > 0:
                    candidate = _CONTINUATION + candidate
                identifier = self._token_to_id.get(candidate)
                if identifier is not None:
                    match = identifier
                    break
                end -= 1
            if match is None:
                # No prefix of the remainder is in the vocabulary: the whole
                # word is unknown, not the partial match collected so far.
                return [UNK_ID]
            pieces.append(match)
            start = end
        return pieces
