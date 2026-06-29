"""Tests for the dev-console debug router (retrieval proxy)."""

from __future__ import annotations

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.servers.web.debug_router import create_debug_router


def _client(handler) -> TestClient:
    """Mount the debug router with an injected httpx client backed by *handler*."""
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    app = FastAPI()
    app.include_router(
        create_debug_router(
            search_url="http://retrieval:8001/retrieve", http_client=http_client
        )
    )
    return TestClient(app)


def test_retrieval_proxy_forwards_to_internal_search_and_returns_results():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "results": [{"doc_id": "d1", "title": "T", "text": "b", "score": 0.9}],
                "retrieval_mode": "sparse",
                "executed_queries": ["vector database"],
                "latency_ms": 1.2,
            },
        )

    client = _client(handler)
    resp = client.post(
        "/api/debug/retrieval/sparse",
        json={"query": "vector database", "top_k": 5},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieval_mode"] == "sparse"
    assert body["results"][0]["doc_id"] == "d1"
    # Proxy must derive the per-mode endpoint from the /retrieve base URL.
    assert captured["url"] == "http://retrieval:8001/internal/search/sparse"


def test_hybrid_proxy_forwards_tuning_knobs():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"retrieval_mode": "hybrid", "results": []})

    client = _client(handler)
    resp = client.post(
        "/api/debug/retrieval/hybrid",
        json={
            "query": "q",
            "top_k": 5,
            "rrf_k": 30,
            "mmr_lambda": 0.7,
            "over_fetch": 3,
        },
    )

    assert resp.status_code == 200
    assert captured["body"] == {
        "query": "q",
        "top_k": 5,
        "rerank": False,
        "rrf_k": 30,
        "mmr_lambda": 0.7,
        "over_fetch": 3,
    }


def test_proxy_forwards_rerank_flag():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"retrieval_mode": "sparse", "results": []})

    client = _client(handler)
    resp = client.post(
        "/api/debug/retrieval/sparse",
        json={"query": "q", "top_k": 5, "rerank": True},
    )

    assert resp.status_code == 200
    assert captured["body"]["rerank"] is True


def test_proxy_rerank_defaults_false():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"retrieval_mode": "sparse", "results": []})

    client = _client(handler)
    client.post("/api/debug/retrieval/sparse", json={"query": "q", "top_k": 5})
    assert captured["body"]["rerank"] is False


def test_proxy_passes_through_503_when_dense_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "Dense search not configured"})

    client = _client(handler)
    resp = client.post("/api/debug/retrieval/dense", json={"query": "q", "top_k": 5})

    assert resp.status_code == 503
    assert "Dense" in resp.json()["detail"]


def test_proxy_passes_through_404_when_endpoint_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    client = _client(handler)
    resp = client.post("/api/debug/retrieval/sparse", json={"query": "q", "top_k": 5})

    assert resp.status_code == 404


def test_unknown_mode_rejected_without_calling_upstream():
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    client = _client(handler)
    resp = client.post("/api/debug/retrieval/bogus", json={"query": "q", "top_k": 5})

    assert resp.status_code == 404
    assert called is False
