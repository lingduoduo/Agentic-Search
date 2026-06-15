"""Tests for mid-execution fallbacks in _run_auto_routed."""

from __future__ import annotations

from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from src.context.models import (
    AnswerGenerationResult,
    SearchContextBundle,
    PromptBundle,
    ContextDocument,
)
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app
from src.agents.base import AgentLoopOutput


def _make_answer_result(answer: str = "ok") -> AnswerGenerationResult:
    return AnswerGenerationResult(
        answer=answer,
        citations=[],
        context=SearchContextBundle(query="q", documents=[]),
        prompt=PromptBundle(system="", user="", messages=[]),
    )


def _make_loop_output(final_answer: str | None = "tool answer") -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        final_answer=final_answer,
    )


# --- Intent model fallbacks ---


def test_tool_loop_raises_reroutes_to_tier2(monkeypatch, tmp_path):
    """ToolAgentLoop.run raises → falls back to answer_with_retrieval via Tier 2/3."""
    monkeypatch.setattr(
        "src.agents.tool_calling.ToolAgentLoop.run",
        AsyncMock(side_effect=RuntimeError("OOM")),
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        AsyncMock(return_value=_make_answer_result("tier2 answer")),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        # "explain" → rule-based classifies as chat → goes to answer_with_retrieval
        response = client.post("/api/agent", json={"query": "explain what is FAISS"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "chat"
    assert data["answer"] == "tier2 answer"


def test_tool_loop_empty_output_reroutes(monkeypatch, tmp_path):
    """ToolAgentLoop returns final_answer=None → falls back to Tier 2/3."""
    monkeypatch.setattr(
        "src.agents.tool_calling.ToolAgentLoop.run",
        AsyncMock(return_value=_make_loop_output(final_answer=None)),
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        AsyncMock(return_value=_make_answer_result("fallback answer")),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        # "explain" → rule-based classifies as chat → goes to answer_with_retrieval
        response = client.post("/api/agent", json={"query": "explain FAISS"})
    assert response.status_code == 200
    assert response.json()["intent"] == "chat"


# --- Search fallbacks ---


def test_hybrid_search_fails_falls_back_to_rag_without_context(monkeypatch, tmp_path):
    """_run_hybrid_search raises → answer_with_retrieval called with top_k=0."""
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_hybrid_search",
        AsyncMock(side_effect=ConnectionError("retrieval down")),
    )
    rag_call_kwargs: dict = {}

    async def fake_rag(q, *, llm, chat_history, search_url, top_k, filters):
        rag_call_kwargs["top_k"] = top_k
        return _make_answer_result("rag fallback")

    monkeypatch.setattr("src.internal.servers.web.app.answer_with_retrieval", fake_rag)
    # No llm → rule-based classifier used. "find onboarding doc" → is_search=True
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "find onboarding doc"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "chat"
    assert rag_call_kwargs.get("top_k") == 0


def test_hybrid_search_and_rag_both_fail_returns_502(monkeypatch, tmp_path):
    """_run_hybrid_search raises, answer_with_retrieval raises, _run_direct_search raises → 502."""
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_hybrid_search",
        AsyncMock(side_effect=ConnectionError("down")),
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        AsyncMock(side_effect=RuntimeError("llm down")),
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_direct_search",
        AsyncMock(side_effect=ConnectionError("still down")),
    )
    # No llm → rule-based classifier. "find doc" → is_search=True
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "find doc"})
    assert response.status_code == 502
    assert "retrieval also unavailable" in response.json()["detail"].lower()


# --- RAG fallbacks ---


def test_answer_with_retrieval_fails_falls_back_to_raw_docs(monkeypatch, tmp_path):
    """answer_with_retrieval raises → _run_direct_search returns raw docs → 200 with search intent."""
    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        AsyncMock(side_effect=RuntimeError("llm down")),
    )
    raw_doc = ContextDocument(
        id="D1", title="Doc", content="content", url=None, score=0.9, metadata={}
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_direct_search",
        AsyncMock(return_value=[raw_doc, raw_doc, raw_doc]),
    )
    # No llm → rule-based classifier. "explain FAISS" → is_search=False → chat path
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "explain FAISS"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "search"
    assert len(data["documents"]) == 3


def test_answer_with_retrieval_and_search_both_fail_returns_502(monkeypatch, tmp_path):
    """answer_with_retrieval raises, _run_direct_search raises → 502."""
    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        AsyncMock(side_effect=RuntimeError("llm down")),
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_direct_search",
        AsyncMock(side_effect=ConnectionError("retrieval down")),
    )
    # No llm → rule-based classifier. "explain FAISS" → is_search=False → chat path
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "explain FAISS"})
    assert response.status_code == 502
    assert "retrieval also unavailable" in response.json()["detail"].lower()
