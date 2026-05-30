"""Embedding facade for indexing chunks."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .index_builder import EmbeddingFn
from .index_builder import deterministic_embedding_fn
from .index_builder import embed_chunks
from .index_builder import embed_chunks_with_failure_handling
from .indexing_heartbeat import IndexingHeartbeatInterface
from .models import EmbeddedChunk
from .models import EmbeddingConfig
from .models import IndexChunk


class IndexingEmbedder(Protocol):
    """Converts index chunks into embedded chunks."""

    def embed_chunks(self, chunks: list[IndexChunk]) -> list[EmbeddedChunk]: ...


class DefaultIndexingEmbedder:
    """Small embedding wrapper used by local indexing flows and tests."""

    def __init__(
        self,
        embedding_fn: EmbeddingFn | None = None,
        config: EmbeddingConfig | None = None,
        *,
        callback: IndexingHeartbeatInterface | None = None,
    ) -> None:
        self.embedding_fn = embedding_fn or deterministic_embedding_fn()
        self.config = config or EmbeddingConfig()
        self.callback = callback

    def embed_chunks(self, chunks: list[IndexChunk]) -> list[EmbeddedChunk]:
        return embed_chunks(
            chunks,
            embedding_fn=self.embedding_fn,
            config=self.config,
            callback=self.callback,
        )


def numpy_embedding_fn(vectors: list[list[float]]) -> EmbeddingFn:
    """Build a deterministic test embedder from fixed vectors."""

    matrix = np.asarray(vectors, dtype=np.float32)

    def embed(texts: list[str]) -> np.ndarray:
        if len(texts) > len(matrix):
            raise ValueError("Not enough fixed vectors for requested texts.")
        return matrix[: len(texts)]

    return embed


__all__ = [
    "DefaultIndexingEmbedder",
    "IndexingEmbedder",
    "embed_chunks",
    "embed_chunks_with_failure_handling",
    "numpy_embedding_fn",
]
