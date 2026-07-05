from __future__ import annotations

from src.internal.servers.web.request_capture_store import RequestCaptureStore


def _snap(rid: str) -> dict:
    return {
        "request_id": rid,
        "query": f"q-{rid}",
        "created_at": 1.0,
        "route": "chat",
        "stages": [
            {
                "stage": "intent",
                "label": "x",
                "timestamp": 0,
                "duration_ms": 1,
                "payload": {},
            }
        ],
    }


def test_put_get_roundtrip():
    store = RequestCaptureStore(max_size=3)
    store.put(_snap("a"))
    assert store.get("a")["query"] == "q-a"
    assert store.get("missing") is None


def test_list_is_newest_first_with_summary_fields():
    store = RequestCaptureStore(max_size=3)
    store.put(_snap("a"))
    store.put(_snap("b"))
    listed = store.list()
    assert [r["request_id"] for r in listed] == ["b", "a"]
    assert listed[0] == {
        "request_id": "b",
        "query": "q-b",
        "created_at": 1.0,
        "route": "chat",
        "stage_count": 1,
    }


def test_evicts_beyond_max_size():
    store = RequestCaptureStore(max_size=2)
    store.put(_snap("a"))
    store.put(_snap("b"))
    store.put(_snap("c"))
    assert store.get("a") is None
    assert [r["request_id"] for r in store.list()] == ["c", "b"]
