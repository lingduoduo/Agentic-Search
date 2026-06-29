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


class _StubPipeline:
    def __init__(self, bundle):
        self._bundle = bundle

    def transform(self, query, filters=None):
        return self._bundle


def test_query_transform_returns_variants_and_filters(monkeypatch):
    import src.internal.servers.web.debug_router as mod
    from src.context.query_transform import TransformedQueryBundle

    bundle = TransformedQueryBundle(
        original="vector db",
        sub_queries=["what is a vector database", "how do embeddings index"],
        merged_filters={"year": 2024},
    )
    monkeypatch.setattr(
        mod,
        "build_query_transform_pipeline_from_env",
        lambda llm: _StubPipeline(bundle),
    )
    client = _client(lambda r: httpx.Response(200, json={}))
    resp = client.post("/api/debug/query-transform", json={"query": "vector db"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] is True
    assert "what is a vector database" in body["variants"]
    assert body["merged_filters"] == {"year": 2024}
    assert body["legs"]["sub_queries"] == [
        "what is a vector database",
        "how do embeddings index",
    ]


def test_query_transform_no_pipeline_is_inactive(monkeypatch):
    import src.internal.servers.web.debug_router as mod

    monkeypatch.setattr(
        mod, "build_query_transform_pipeline_from_env", lambda llm: None
    )
    client = _client(lambda r: httpx.Response(200, json={}))
    resp = client.post("/api/debug/query-transform", json={"query": "vector db"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] is False
    assert body["variants"] == ["vector db"]


def test_health_reports_retrieval_up_and_web_self():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"})

    client = _client(handler)
    resp = client.get("/api/debug/health")
    assert resp.status_code == 200
    servers = {s["name"]: s for s in resp.json()["servers"]}
    assert servers["retrieval"]["status"] == "up"
    assert servers["web"]["status"] == "up"


def test_health_reports_retrieval_down_when_unreachable_no_500():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _client(handler)
    resp = client.get("/api/debug/health")
    assert resp.status_code == 200  # never raises
    servers = {s["name"]: s for s in resp.json()["servers"]}
    assert servers["retrieval"]["status"] == "down"
    assert servers["web"]["status"] == "up"


def test_health_reports_retrieval_down_on_non_200():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "unhealthy"})

    client = _client(handler)
    resp = client.get("/api/debug/health")
    servers = {s["name"]: s for s in resp.json()["servers"]}
    assert servers["retrieval"]["status"] == "down"


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
