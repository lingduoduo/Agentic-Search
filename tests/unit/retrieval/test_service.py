"""Tests for RetrievalService."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.service import RetrievalService


def _make_result(doc_id: str, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="T", text="body", url=None, score=score)


def test_search_delegates_to_backend_sparse():
    backend = MagicMock()
    backend.search_sparse.return_value = [_make_result("d1")]
    service = RetrievalService(backend)

    results, mode = service.search("procurement", top_k=5)

    backend.search_sparse.assert_called_once_with("procurement", top_k=5)
    assert mode == "sparse"
    assert len(results) == 1
    assert results[0].doc_id == "d1"


def test_search_returns_empty_list_on_no_results():
    backend = MagicMock()
    backend.search_sparse.return_value = []
    service = RetrievalService(backend)

    results, mode = service.search("nothing", top_k=5)

    assert results == []
    assert mode == "sparse"


def test_from_env_raises_on_unknown_backend(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_BACKEND", "pinecone")
    with pytest.raises(ValueError, match="Unknown RETRIEVAL_BACKEND"):
        RetrievalService.from_env()


def test_from_env_local_requires_index_path(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_BACKEND", "local")
    monkeypatch.delenv("BM25_INDEX_PATH", raising=False)
    with pytest.raises(KeyError):
        RetrievalService.from_env()
