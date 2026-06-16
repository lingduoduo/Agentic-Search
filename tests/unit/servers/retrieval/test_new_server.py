"""Tests for the new retrieval service FastAPI app (server.py)."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.service import RetrievalService
from src.internal.servers.retrieval.server import create_app


def _make_service(
    results: list[RetrievalResult], mode: str = "sparse"
) -> RetrievalService:
    svc = MagicMock(spec=RetrievalService)
    svc.search.return_value = (results, mode)
    return svc


def _result(doc_id: str = "d1", score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id, title="Title", text="body", url="https://x.com", score=score
    )


def test_health_returns_ok():
    client = TestClient(create_app(_make_service([])))
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "backend" in data


def test_search_returns_results():
    svc = _make_service([_result("d1", 0.9), _result("d2", 0.7)])
    client = TestClient(create_app(svc))

    resp = client.post("/search", json={"query": "procurement", "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["retrieval_mode"] == "sparse"
    assert data["executed_queries"] == ["procurement"]
    assert len(data["results"]) == 2
    assert data["results"][0]["doc_id"] == "d1"
    assert "latency_ms" in data


def test_search_calls_service_with_top_k():
    svc = _make_service([])
    client = TestClient(create_app(svc))

    client.post("/search", json={"query": "vector search", "top_k": 10})
    svc.search.assert_called_once_with("vector search", top_k=10, filters=None)


def test_search_rejects_empty_query():
    client = TestClient(create_app(_make_service([])))
    resp = client.post("/search", json={"query": "", "top_k": 5})
    assert resp.status_code == 422


def test_search_default_top_k_is_5():
    svc = _make_service([])
    client = TestClient(create_app(svc))

    client.post("/search", json={"query": "anything"})
    svc.search.assert_called_once_with("anything", top_k=5, filters=None)


def test_health_response_has_api_version_header():
    client = TestClient(create_app(_make_service([])))
    resp = client.get("/health")
    assert resp.headers.get("Retrieval-API-Version") == "1.0"


def test_search_response_has_api_version_header():
    client = TestClient(create_app(_make_service([_result()])))
    resp = client.post("/search", json={"query": "anything"})
    assert resp.headers.get("Retrieval-API-Version") == "1.0"
