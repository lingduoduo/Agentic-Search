"""Tests for internal eval endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.service import RetrievalService
from src.internal.servers.retrieval.eval_router import create_eval_router


def _make_backend(sparse=None, dense=None):
    backend = MagicMock()
    backend.search_sparse.return_value = sparse or []
    if dense is None:
        backend.search_dense.side_effect = NotImplementedError("no dense")
    else:
        backend.search_dense.return_value = dense
    return backend


def _result(doc_id: str = "d1") -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="T", text="b", url=None, score=0.9)


def _app_with_router(backend) -> TestClient:
    svc = RetrievalService(backend)
    app = FastAPI()
    app.include_router(create_eval_router(svc, require_admin=None))
    return TestClient(app)


def test_sparse_endpoint_returns_results():
    client = _app_with_router(_make_backend(sparse=[_result("s1")]))
    resp = client.post("/internal/search/sparse", json={"query": "q", "top_k": 5})
    assert resp.status_code == 200
    assert resp.json()["results"][0]["doc_id"] == "s1"
    assert resp.json()["retrieval_mode"] == "sparse"


def test_dense_endpoint_returns_results():
    client = _app_with_router(_make_backend(dense=[_result("d1")]))
    resp = client.post("/internal/search/dense", json={"query": "q", "top_k": 5})
    assert resp.status_code == 200
    assert resp.json()["results"][0]["doc_id"] == "d1"
    assert resp.json()["retrieval_mode"] == "dense"


def test_dense_endpoint_503_when_not_configured():
    client = _app_with_router(_make_backend(dense=None))
    resp = client.post("/internal/search/dense", json={"query": "q", "top_k": 5})
    assert resp.status_code == 503


def test_hybrid_endpoint_accepts_tuning_params():
    client = _app_with_router(
        _make_backend(sparse=[_result("s1")], dense=[_result("d1")])
    )
    resp = client.post(
        "/internal/search/hybrid",
        json={
            "query": "q",
            "top_k": 5,
            "rrf_k": 30,
            "mmr_lambda": 0.7,
            "over_fetch": 3,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["retrieval_mode"] == "hybrid"


def test_graph_endpoint_returns_mode_graph():
    client = _app_with_router(_make_backend(sparse=[_result("g1")]))
    resp = client.post(
        "/internal/search/graph",
        json={"query": "FAISS dense retrieval", "top_k": 5},
    )
    assert resp.status_code == 200
    assert resp.json()["retrieval_mode"] == "graph"
    assert len(resp.json()["results"]) >= 1


def test_graph_endpoint_accepts_tuning_params():
    client = _app_with_router(_make_backend(sparse=[_result("g2")]))
    resp = client.post(
        "/internal/search/graph",
        json={"query": "FAISS", "top_k": 3, "initial_k": 2, "max_entity_queries": 1},
    )
    assert resp.status_code == 200
    assert resp.json()["retrieval_mode"] == "graph"


def test_graph_endpoint_empty_corpus_returns_empty():
    client = _app_with_router(_make_backend(sparse=[]))
    resp = client.post(
        "/internal/search/graph",
        json={"query": "nothing here", "top_k": 5},
    )
    assert resp.status_code == 200
    assert resp.json()["results"] == []


class _ReverseReranker:
    """Stub reranker that reverses order — proves reranking actually ran."""

    def rerank(self, query, results, top_k):
        return list(reversed(results))[:top_k]


def test_rerank_true_applies_reranker_and_tags_mode(monkeypatch):
    import src.internal.servers.retrieval.eval_router as mod

    monkeypatch.setattr(mod, "build_reranker_from_env", lambda: _ReverseReranker())
    client = _app_with_router(
        _make_backend(sparse=[_result("a"), _result("b"), _result("c")])
    )
    resp = client.post(
        "/internal/search/sparse", json={"query": "q", "top_k": 5, "rerank": True}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieval_mode"] == "sparse+reranked"
    # same candidate set, reordered
    assert [r["doc_id"] for r in body["results"]] == ["c", "b", "a"]


def test_rerank_true_without_reranker_is_noop(monkeypatch):
    import src.internal.servers.retrieval.eval_router as mod

    monkeypatch.setattr(mod, "build_reranker_from_env", lambda: None)
    client = _app_with_router(_make_backend(sparse=[_result("a"), _result("b")]))
    resp = client.post(
        "/internal/search/sparse", json={"query": "q", "top_k": 5, "rerank": True}
    )
    assert resp.status_code == 200
    body = resp.json()
    # no reranker configured → order unchanged, mode NOT tagged (visible "no reranker")
    assert body["retrieval_mode"] == "sparse"
    assert [r["doc_id"] for r in body["results"]] == ["a", "b"]


def test_rerank_default_false_unchanged():
    client = _app_with_router(_make_backend(sparse=[_result("a")]))
    resp = client.post("/internal/search/sparse", json={"query": "q", "top_k": 5})
    assert resp.status_code == 200
    assert resp.json()["retrieval_mode"] == "sparse"
