"""Tests for POST /api/feedback router."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.db.store import AgenticSearchStore
from src.internal.servers.retrieval.feedback_router import create_feedback_router


def _app(db: AgenticSearchStore) -> TestClient:
    app = FastAPI()
    app.include_router(create_feedback_router(db))
    return TestClient(app)


def test_feedback_thumbs_up_persisted():
    db = AgenticSearchStore(":memory:")
    client = _app(db)

    resp = client.post(
        "/api/feedback", json={"session_id": "s1", "signal": "thumbs_up"}
    )

    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert db.get_feedback_summary()["rated_queries"] == 1


def test_feedback_thumbs_down_persisted():
    db = AgenticSearchStore(":memory:")
    client = _app(db)

    resp = client.post(
        "/api/feedback", json={"session_id": "s1", "signal": "thumbs_down"}
    )

    assert resp.status_code == 200
    summary = db.get_feedback_summary()
    assert summary["thumbs_up_rate"] == 0.0


def test_feedback_invalid_signal_rejected():
    db = AgenticSearchStore(":memory:")
    client = _app(db)

    resp = client.post("/api/feedback", json={"session_id": "s1", "signal": "meh"})

    assert resp.status_code == 422


def test_feedback_multiple_signals_accumulate():
    db = AgenticSearchStore(":memory:")
    client = _app(db)

    client.post("/api/feedback", json={"session_id": "s1", "signal": "thumbs_up"})
    client.post("/api/feedback", json={"session_id": "s2", "signal": "thumbs_up"})
    client.post("/api/feedback", json={"session_id": "s3", "signal": "thumbs_down"})

    summary = db.get_feedback_summary()
    assert summary["rated_queries"] == 3
