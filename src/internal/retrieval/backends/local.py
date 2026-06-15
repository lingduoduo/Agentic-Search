"""Local backend: wraps Pyserini SparseRetriever (BM25) and DenseRetriever (FAISS)."""

from __future__ import annotations

from src.internal.document_index.retrieval import (
    DenseRetriever,
    DenseRetrieverConfig,
    SparseRetriever,
    SparseRetrieverConfig,
)

from .base import RetrievalBackend, RetrievalResult


def _make_sparse_retriever(config: SparseRetrieverConfig) -> SparseRetriever:
    """Thin factory — exists so tests can monkeypatch it."""
    return SparseRetriever(config)


def _make_dense_retriever(config: DenseRetrieverConfig) -> DenseRetriever:
    """Thin factory — exists so tests can monkeypatch it."""
    return DenseRetriever(config)


def _row_to_result(row: dict) -> RetrievalResult:
    """Convert a raw retriever row dict into a RetrievalResult."""
    doc = row.get("document", {})
    text: str = doc.get("text") or doc.get("contents") or ""
    # Corpus stores chunks as '"Title"\nBody...' — strip the quoted title prefix.
    if text.startswith('"'):
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) > 1 else text
    return RetrievalResult(
        doc_id=str(doc.get("id", "")),
        title=str(doc.get("title", "")),
        text=text,
        url=doc.get("url"),
        score=float(row.get("score", 0.0)),
    )


class LocalBackend(RetrievalBackend):
    """Backend that retrieves from a local Pyserini BM25 index and optional FAISS index."""

    def __init__(
        self,
        sparse_config: SparseRetrieverConfig,
        dense_config: DenseRetrieverConfig | None = None,
    ) -> None:
        self._sparse = _make_sparse_retriever(sparse_config)
        self._dense = _make_dense_retriever(dense_config) if dense_config else None

    def search_sparse(self, query: str, top_k: int) -> list[RetrievalResult]:
        rows = self._sparse.retrieve([query], topk=top_k)
        return [_row_to_result(r) for r in rows[0]]

    def search_dense(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self._dense is None:
            raise NotImplementedError(
                "Dense search not configured — set DENSE_MODEL_PATH env var"
            )
        rows = self._dense.retrieve([query], topk=top_k)
        return [_row_to_result(r) for r in rows[0]]
