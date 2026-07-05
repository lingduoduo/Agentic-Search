from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.servers.web.debug_router import create_debug_router
from src.internal.servers.web.request_capture_store import RequestCaptureStore


def _app_with_store(store: RequestCaptureStore) -> FastAPI:
    app = FastAPI()
    app.state.request_captures = store
    app.include_router(create_debug_router(search_url="http://x/retrieve"))
    return app


def _snap(rid: str) -> dict:
    return {
        "request_id": rid,
        "query": f"q-{rid}",
        "created_at": 1.0,
        "route": "chat",
        "route_degraded": None,
        "total_ms": 5.0,
        "stages": [
            {
                "stage": "intent",
                "label": "x",
                "timestamp": 0.0,
                "duration_ms": 1.0,
                "payload": {"raw": "chat"},
            }
        ],
    }


def test_list_requests_newest_first():
    store = RequestCaptureStore()
    store.put(_snap("a"))
    store.put(_snap("b"))
    client = TestClient(_app_with_store(store))
    body = client.get("/api/debug/requests").json()
    assert [r["request_id"] for r in body["requests"]] == ["b", "a"]
    assert body["requests"][0]["stage_count"] == 1


def test_get_request_returns_full_snapshot():
    store = RequestCaptureStore()
    store.put(_snap("a"))
    client = TestClient(_app_with_store(store))
    body = client.get("/api/debug/request/a").json()
    assert body["stages"][0]["payload"] == {"raw": "chat"}


def test_get_missing_request_is_404():
    client = TestClient(_app_with_store(RequestCaptureStore()))
    assert client.get("/api/debug/request/nope").status_code == 404
