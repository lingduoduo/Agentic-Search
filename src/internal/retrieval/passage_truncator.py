from __future__ import annotations

import os


class PassageTruncator:
    """Whitespace-token truncation for reranker passages."""

    def __init__(self, max_tokens: int = 512) -> None:
        self._max = max_tokens

    def truncate(self, text: str) -> str:
        if self._max == 0 or not text:
            return text
        tokens = text.split()
        if len(tokens) <= self._max:
            return text
        return " ".join(tokens[: self._max])

    @staticmethod
    def from_env() -> PassageTruncator:
        return PassageTruncator(
            max_tokens=int(os.environ.get("RERANKER_MAX_TOKENS", "512"))
        )
