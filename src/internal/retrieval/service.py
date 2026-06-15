"""RetrievalService: selects backend from env and exposes search()."""

from __future__ import annotations

import os

from .backends.base import RetrievalBackend, RetrievalResult


def _build_local_backend() -> RetrievalBackend:
    from src.internal.document_index.retrieval import SparseRetrieverConfig

    from .backends.local import LocalBackend

    config = SparseRetrieverConfig(
        index_path=os.environ["BM25_INDEX_PATH"],
        corpus_path=os.environ.get("BM25_CORPUS_PATH", "data/corpus.jsonl"),
        topk=int(os.environ.get("BM25_TOP_K", "20")),
    )
    return LocalBackend(config)


def _build_backend() -> RetrievalBackend:
    name = os.environ.get("RETRIEVAL_BACKEND", "local").lower()
    if name == "local":
        return _build_local_backend()
    raise ValueError(
        f"Unknown RETRIEVAL_BACKEND: {name!r}. Supported values: local"
        " (opensearch and weaviate added in M3)"
    )


class RetrievalService:
    def __init__(self, backend: RetrievalBackend) -> None:
        self._backend = backend

    @classmethod
    def from_env(cls) -> "RetrievalService":
        """Construct service from environment variables."""
        return cls(_build_backend())

    def search(self, query: str, top_k: int = 5) -> tuple[list[RetrievalResult], str]:
        """Return (results, retrieval_mode).

        retrieval_mode is 'sparse' in M1; becomes 'hybrid' in M2.
        """
        results = self._backend.search_sparse(query, top_k=top_k)
        return results, "sparse"
