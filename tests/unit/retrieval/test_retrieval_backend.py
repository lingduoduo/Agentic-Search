"""Tests for RetrievalResult and RetrievalBackend ABC."""

from __future__ import annotations

import pytest

from src.internal.retrieval.backends.base import RetrievalBackend, RetrievalResult


def test_retrieval_result_defaults():
    r = RetrievalResult(doc_id="d1", title="T", text="body", url=None, score=0.5)
    assert r.metadata == {}


def test_retrieval_result_stores_fields():
    r = RetrievalResult(
        doc_id="d1",
        title="Title",
        text="body",
        url="https://x.com",
        score=0.9,
        metadata={"src": "corpus"},
    )
    assert r.doc_id == "d1"
    assert r.score == 0.9
    assert r.url == "https://x.com"
    assert r.metadata == {"src": "corpus"}


def test_retrieval_backend_is_abstract():
    """Cannot instantiate RetrievalBackend directly."""
    with pytest.raises(TypeError):
        RetrievalBackend()  # type: ignore[abstract]


class _ConcreteBackend(RetrievalBackend):
    def search_sparse(self, query: str, top_k: int) -> list[RetrievalResult]:
        return []

    def search_dense(self, query: str, top_k: int) -> list[RetrievalResult]:
        raise NotImplementedError


def test_concrete_backend_instantiates():
    b = _ConcreteBackend()
    assert b.search_sparse("q", 5) == []


# --- LocalBackend ---

from unittest.mock import MagicMock  # noqa: E402

from src.internal.retrieval.backends.local import LocalBackend, _row_to_result  # noqa: E402


def _fake_sparse_retriever(rows: list[dict]) -> MagicMock:
    m = MagicMock()
    m.retrieve.return_value = [rows]
    return m


def test_row_to_result_standard_keys():
    row = {
        "document": {
            "id": "d1",
            "title": "T1",
            "contents": "body text",
            "url": "https://x.com",
        },
        "score": 0.8,
    }
    r = _row_to_result(row)
    assert r.doc_id == "d1"
    assert r.title == "T1"
    assert r.text == "body text"
    assert r.url == "https://x.com"
    assert r.score == 0.8


def test_row_to_result_quoted_title_prefix_stripped():
    row = {
        "document": {"id": "d2", "title": "T2", "contents": '"T2"\nActual body'},
        "score": 0.5,
    }
    r = _row_to_result(row)
    assert r.text == "Actual body"


def test_local_backend_search_sparse(monkeypatch):
    import src.internal.retrieval.backends.local as local_mod

    rows = [
        {
            "document": {"id": "d1", "title": "T1", "contents": "body", "url": None},
            "score": 0.9,
        },
        {
            "document": {"id": "d2", "title": "T2", "contents": "text", "url": None},
            "score": 0.7,
        },
    ]
    fake = _fake_sparse_retriever(rows)
    monkeypatch.setattr(local_mod, "_make_sparse_retriever", lambda cfg: fake)

    from src.internal.document_index.retrieval import SparseRetrieverConfig

    backend = LocalBackend(SparseRetrieverConfig(index_path="x", corpus_path="y"))
    results = backend.search_sparse("retrieval", top_k=5)

    assert len(results) == 2
    assert results[0].doc_id == "d1"
    assert results[0].score == 0.9
    assert results[1].doc_id == "d2"
    fake.retrieve.assert_called_once_with(["retrieval"], topk=5)


def test_local_backend_search_dense_raises(monkeypatch):
    import src.internal.retrieval.backends.local as local_mod

    monkeypatch.setattr(local_mod, "_make_sparse_retriever", lambda cfg: MagicMock())

    from src.internal.document_index.retrieval import SparseRetrieverConfig

    backend = LocalBackend(SparseRetrieverConfig(index_path="x", corpus_path="y"))
    with pytest.raises(NotImplementedError, match="Dense search not configured"):
        backend.search_dense("q", 5)


def _fake_dense_retriever(rows: list[dict]) -> MagicMock:
    m = MagicMock()
    m.retrieve.return_value = [rows]
    return m


def test_local_backend_search_dense(monkeypatch):
    import src.internal.retrieval.backends.local as local_mod

    from src.internal.document_index.retrieval import (
        DenseRetrieverConfig,
        SparseRetrieverConfig,
    )

    dense_rows = [
        {
            "document": {
                "id": "d3",
                "title": "T3",
                "contents": "dense body",
                "url": None,
            },
            "score": 0.95,
        },
    ]
    monkeypatch.setattr(local_mod, "_make_sparse_retriever", lambda cfg: MagicMock())
    monkeypatch.setattr(
        local_mod,
        "_make_dense_retriever",
        lambda cfg: _fake_dense_retriever(dense_rows),
    )

    backend = LocalBackend(
        SparseRetrieverConfig(index_path="x", corpus_path="y"),
        dense_config=DenseRetrieverConfig.for_e5_base_v2(
            index_path="z", corpus_path="y"
        ),
    )
    results = backend.search_dense("embedding", top_k=5)
    assert len(results) == 1
    assert results[0].doc_id == "d3"
    assert results[0].score == pytest.approx(0.95)


def test_local_backend_search_dense_raises_when_not_configured(monkeypatch):
    import src.internal.retrieval.backends.local as local_mod

    from src.internal.document_index.retrieval import SparseRetrieverConfig

    monkeypatch.setattr(local_mod, "_make_sparse_retriever", lambda cfg: MagicMock())

    backend = LocalBackend(SparseRetrieverConfig(index_path="x", corpus_path="y"))
    with pytest.raises(NotImplementedError, match="Dense search not configured"):
        backend.search_dense("q", 5)
