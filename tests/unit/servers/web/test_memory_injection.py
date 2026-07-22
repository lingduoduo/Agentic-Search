"""End-to-end: the /api/agent dispatch threads the user's memory into the
answer path when AGENTIC_SEARCH_MEMORY_INJECTION is on."""

from fastapi.testclient import TestClient

from src.context.models import (
    AnswerGenerationResult,
    PromptBundle,
    SearchContextBundle,
)
from src.internal.db.models import UserRecord
from src.internal.db.store import AgenticSearchStore
from src.internal.servers.web import app as web_app
from src.internal.servers.web.app import SearchExperienceSettings


def _capturing_awr(captured: dict):
    async def fake_answer_with_retrieval(question, **kwargs):
        captured["user_memory"] = kwargs.get("user_memory")
        ctx = SearchContextBundle(query=question, documents=[])
        return AnswerGenerationResult(
            answer="ok",
            citations=[],
            context=ctx,
            prompt=PromptBundle(system="", user="", messages=[]),
        )

    return fake_answer_with_retrieval


def test_memory_injected_when_flag_on(monkeypatch):
    store = AgenticSearchStore(":memory:")
    store.upsert_user(UserRecord(id="u1"))
    store.add_user_memory("u1", "User is allergic to peanuts")
    captured: dict = {}
    monkeypatch.setattr(web_app, "answer_with_retrieval", _capturing_awr(captured))

    app = web_app.create_web_app(
        SearchExperienceSettings(memory_injection=True), store=store
    )
    resp = TestClient(app).post(
        "/api/agent",
        json={"query": "Recommend Thai food", "mode": "chat_once", "user_id": "u1"},
    )
    assert resp.status_code == 200
    assert captured["user_memory"] is not None
    assert "allergic to peanuts" in captured["user_memory"]
    store.close()


def test_memory_not_injected_when_flag_off(monkeypatch):
    store = AgenticSearchStore(":memory:")
    store.upsert_user(UserRecord(id="u1"))
    store.add_user_memory("u1", "User is allergic to peanuts")
    captured: dict = {}
    monkeypatch.setattr(web_app, "answer_with_retrieval", _capturing_awr(captured))

    app = web_app.create_web_app(
        SearchExperienceSettings(memory_injection=False), store=store
    )
    resp = TestClient(app).post(
        "/api/agent",
        json={"query": "Recommend Thai food", "mode": "chat_once", "user_id": "u1"},
    )
    assert resp.status_code == 200
    assert captured["user_memory"] is None
    store.close()


def test_flag_reads_env(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_MEMORY_INJECTION", "1")
    assert SearchExperienceSettings.from_app_settings().memory_injection is True
    monkeypatch.setenv("AGENTIC_SEARCH_MEMORY_INJECTION", "")
    assert SearchExperienceSettings.from_app_settings().memory_injection is False
