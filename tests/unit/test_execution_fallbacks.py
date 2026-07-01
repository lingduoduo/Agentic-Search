"""Tests for mid-execution fallbacks in the 3-way agentic router.

The router (`route_query`) picks a strategy; dispatch is capability-aware. The
retrieval-first fallback chain (hybrid -> RAG -> raw docs -> 502) lives in
`_auto_search_pipeline`, reached when SEARCH has no local model or
CHAT has no LLM. These tests force a strategy via `route_query` and assert
the resulting dispatch / fallback behavior.
"""

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
from src.internal.servers.web.intent_routing import RouteStrategy
from src.agents.base import AgentLoopOutput


def _make_answer_result(answer: str = "ok") -> AnswerGenerationResult:
    return AnswerGenerationResult(
        answer=answer,
        citations=[],
        context=SearchContextBundle(query="q", documents=[]),
        prompt=PromptBundle(system="", user="", messages=[]),
    )


def _force_route(monkeypatch, strategy: RouteStrategy) -> None:
    monkeypatch.setattr(
        "src.internal.servers.web.app.route_query", lambda *a, **k: strategy
    )


# --- SEARCH without a local model degrades to the hybrid pipeline ---


def test_search_agent_without_model_runs_hybrid_pipeline(monkeypatch, tmp_path):
    """SEARCH + no local model → hybrid pipeline, intent='search'."""
    from src.internal.servers.web.app import _HybridSearchResult

    async def fake_hybrid(query, **kwargs):
        doc = ContextDocument(id="D1", title="t", content="c", url=None, score=0.0)
        return _HybridSearchResult(executed_queries=[query], documents=[doc])

    _force_route(monkeypatch, RouteStrategy.SEARCH)
    monkeypatch.setattr("src.internal.servers.web.app._run_hybrid_search", fake_hybrid)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "find onboarding doc"})
    assert response.status_code == 200
    assert response.json()["intent"] == "search"


# --- Search fallbacks inside _auto_search_pipeline ---


def test_hybrid_search_fails_falls_back_to_rag_without_context(monkeypatch, tmp_path):
    """hybrid raises → answer_with_retrieval called with top_k=0, intent='chat'."""
    _force_route(monkeypatch, RouteStrategy.SEARCH)
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_hybrid_search",
        AsyncMock(side_effect=ConnectionError("retrieval down")),
    )
    rag_call_kwargs: dict = {}

    async def fake_rag(q, *, llm, chat_history, search_url, top_k, filters):
        rag_call_kwargs["top_k"] = top_k
        return _make_answer_result("rag fallback")

    monkeypatch.setattr("src.internal.servers.web.app.answer_with_retrieval", fake_rag)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "find onboarding doc"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "chat"
    assert rag_call_kwargs.get("top_k") == 0


def test_hybrid_search_and_rag_both_fail_returns_502(monkeypatch, tmp_path):
    """hybrid raises, RAG raises, raw search raises → 502."""
    _force_route(monkeypatch, RouteStrategy.SEARCH)
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
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"), llm=MagicMock()
    )
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "find doc"})
    assert response.status_code == 502
    assert "retrieval also unavailable" in response.json()["detail"].lower()


def test_rag_fails_falls_back_to_raw_docs(monkeypatch, tmp_path):
    """hybrid raises, RAG raises, raw search returns docs → 200 with search intent."""
    _force_route(monkeypatch, RouteStrategy.SEARCH)
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_hybrid_search",
        AsyncMock(side_effect=ConnectionError("down")),
    )
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
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"), llm=MagicMock()
    )
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "find doc"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "search"
    assert len(data["documents"]) == 3


def test_search_unreachable_returns_clear_message(monkeypatch, tmp_path):
    from src.internal.servers.web.app import _HybridSearchResult

    async def fake_hybrid(query, **kwargs):
        return _HybridSearchResult(
            executed_queries=[query], documents=[], status="unreachable"
        )

    _force_route(monkeypatch, RouteStrategy.SEARCH)
    monkeypatch.setattr("src.internal.servers.web.app._run_hybrid_search", fake_hybrid)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "find stuff"})
    data = response.json()
    assert data["intent"] == "search"
    assert "No sources are reachable" in data["answer"]
    assert data["documents"] == []


def test_search_empty_uses_no_results_message(monkeypatch, tmp_path):
    from src.internal.servers.web.app import _HybridSearchResult

    async def fake_hybrid(query, **kwargs):
        return _HybridSearchResult(
            executed_queries=[query], documents=[], status="empty"
        )

    _force_route(monkeypatch, RouteStrategy.SEARCH)
    monkeypatch.setattr("src.internal.servers.web.app._run_hybrid_search", fake_hybrid)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "find stuff"})
    data = response.json()
    assert data["intent"] == "search"
    assert "No sources are reachable" not in data["answer"]
    assert "no results" in data["answer"].lower()


# --- TOOL degrade ---


def test_tool_loop_failure_degrades_to_agentic_rag(monkeypatch, tmp_path):
    """TOOL with a model but the loop raises → degrade to CHAT."""
    from src.agents.search import AgenticRAGResult

    _force_route(monkeypatch, RouteStrategy.TOOL)
    monkeypatch.setattr(
        "src.agents.tool.tool_calling.ToolAgentLoop.run",
        AsyncMock(side_effect=RuntimeError("OOM")),
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.AgenticRAGLoop.run",
        AsyncMock(
            return_value=AgenticRAGResult(
                answer="degraded rag answer",
                citations=[],
                context=SearchContextBundle(query="q", documents=[]),
                rounds_used=1,
            )
        ),
    )
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"), llm=MagicMock()
    )
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        response = client.post("/api/agent", json={"query": "do something"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "chat"
    assert data["answer"] == "degraded rag answer"


def test_tool_loop_empty_output_degrades(monkeypatch, tmp_path):
    """TOOL loop returns an empty answer → degrade to CHAT."""
    from src.agents.search import AgenticRAGResult

    _force_route(monkeypatch, RouteStrategy.TOOL)
    monkeypatch.setattr(
        "src.agents.tool.tool_calling.ToolAgentLoop.run",
        AsyncMock(
            return_value=AgentLoopOutput(
                prompt_ids=[],
                response_ids=[],
                response_mask=[],
                num_turns=1,
                final_answer=None,
            )
        ),
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.AgenticRAGLoop.run",
        AsyncMock(
            return_value=AgenticRAGResult(
                answer="fallback rag",
                citations=[],
                context=SearchContextBundle(query="q", documents=[]),
                rounds_used=1,
            )
        ),
    )
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"), llm=MagicMock()
    )
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        response = client.post("/api/agent", json={"query": "do something"})
    assert response.status_code == 200
    assert response.json()["answer"] == "fallback rag"


# --- Explicit source selection forces a search against that provider ---


def test_explicit_source_forces_search_against_that_provider(monkeypatch, tmp_path):
    """source_provider='serpapi' → route_query returns SEARCH; the chosen
    provider flows through to hybrid search.
    """
    from src.internal.servers.web.app import _HybridSearchResult

    captured: dict = {}

    async def fake_hybrid(query, **kwargs):
        captured["source_provider"] = kwargs.get("source_provider")
        doc = ContextDocument(id="D1", title="t", content="c", url=None, score=0.0)
        return _HybridSearchResult(executed_queries=[query], documents=[doc])

    # No route_query override: explicit_source must drive SEARCH on its own.
    monkeypatch.setattr("src.internal.servers.web.app._run_hybrid_search", fake_hybrid)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post(
            "/api/agent",
            json={"query": "explain how FAISS works", "source_provider": "serpapi"},
        )
    assert response.status_code == 200
    assert response.json()["intent"] == "search"
    assert captured["source_provider"] == "serpapi"


def test_auto_provider_expands_to_internal_and_serpapi():
    from src.internal.servers.web.app import _source_providers_for

    assert _source_providers_for("auto") == ["retrieval", "serpapi"]
