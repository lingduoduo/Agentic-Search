"""Abstract base for all retrieval backends."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    doc_id: str
    title: str
    text: str
    url: str | None
    score: float
    metadata: dict = field(default_factory=dict)


class RetrievalBackend(abc.ABC):
    @abc.abstractmethod
    def search_sparse(self, query: str, top_k: int) -> list[RetrievalResult]:
        """BM25 keyword search. Must be implemented by every backend."""

    @abc.abstractmethod
    def search_dense(self, query: str, top_k: int) -> list[RetrievalResult]:
        """ANN vector search. Raise NotImplementedError if not supported."""
