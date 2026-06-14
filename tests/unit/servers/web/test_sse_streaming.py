"""Unit tests for POST /api/agent/stream (SSE)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.context.models import (
    AnswerGenerationResult,
    ContextDocument,
    PromptBundle,
    SearchContextBundle,
)
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[len("data:") :].strip()
            if payload:
                events.append(json.loads(payload))
    return events


def _answer_result(question: str) -> AnswerGenerationResult:
    doc = ContextDocument(
        id="D1",
        title="T",
        content=f"[D1] Answer to {question}",
        url="https://t.test",
        score=0.9,
    )
    return AnswerGenerationResult(
        answer=f"[D1] Answer to {question}",
        citations=["D1"],
        context=SearchContextBundle(query=question, documents=[doc]),
        prompt=PromptBundle(system="", user="", messages=[]),
    )


def test_stream_endpoint_exists(tmp_path):
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"))
    client = TestClient(app)
    resp = client.post("/api/agent/stream", json={"query": "x", "mode": "chat_once"})
    assert resp.status_code != 404


def test_stream_chat_once_emits_answer_and_done(monkeypatch, tmp_path):
    async def fake_answer(question, *, llm=None, chat_history=None, **kw):
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"))
    client = TestClient(app)

    resp = client.post(
        "/api/agent/stream",
        json={"query": "What is FAISS?", "mode": "chat_once"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = _parse_sse(resp.text)
    types = [e["type"] for e in events]
    assert "answer" in types
    assert "done" in types

    answer_event = next(e for e in events if e["type"] == "answer")
    assert "[D1]" in answer_event["text"]

    done_event = next(e for e in events if e["type"] == "done")
    assert done_event["session_id"]
    assert "D1" in done_event["citations"]


def test_stream_emits_error_event_on_failure(monkeypatch, tmp_path):
    async def bad_answer(question, **kw):
        raise RuntimeError("simulated backend failure")

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", bad_answer
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"))
    client = TestClient(app)

    resp = client.post(
        "/api/agent/stream",
        json={"query": "test", "mode": "chat_once"},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    error_events = [e for e in events if e["type"] == "error"]
    assert error_events


def test_stream_done_event_contains_documents(monkeypatch, tmp_path):
    async def fake_answer(question, **kw):
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"))
    client = TestClient(app)

    resp = client.post(
        "/api/agent/stream",
        json={"query": "test", "mode": "chat_once"},
    )
    events = _parse_sse(resp.text)
    done_event = next(e for e in events if e["type"] == "done")
    assert isinstance(done_event["documents"], list)
    assert len(done_event["documents"]) >= 1
    assert done_event["documents"][0]["title"] == "T"
