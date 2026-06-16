"""Typed configuration for document chunking.

Values are fixed at index time — changing them requires a full re-index.
PRD defaults: chunk_size=512 tokens, chunk_overlap=64 tokens.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ChunkConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap must be non-negative, got {self.chunk_overlap}"
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )

    @classmethod
    def from_env(cls) -> "ChunkConfig":
        return cls(
            chunk_size=int(os.environ.get("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "64")),
        )
