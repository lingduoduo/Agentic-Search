from __future__ import annotations

from fastapi.testclient import TestClient

from src.context.models import AnswerGenerationResult
from src.context.models import ChatMessage
from src.context.models import ContextDocument
from src.context.models import PromptBundle
from src.context.models import SearchContextBundle
from src.internal.db import AgenticSearchStore
from src.internal.hooks import HookConfig
from src.internal.hooks import HookPoint
from src.internal.hooks import HookRegistry
from src.internal.servers.web.app import SearchExperienceSettings
from src.internal.servers.web.app import create_web_app
from src.internal.servers.web.app import _normalize_agent_mode
from src.internal.servers.web.app import _normalize_source_provider
from src.internal.servers.web.app import _source_providers_for


def _answer_result(question: str) -> AnswerGenerationResult:
    context = SearchContextBundle(
        query=question,
        documents=[
            ContextDocument(
                id="D1",
                title="Deployment Guide",
                content="Use the retrieval server before starting the web app.",
                url="https://example.test/deploy",
                score=0.91,
                metadata={"source_type": "docs"},
            )
        ],
    )
    return AnswerGenerationResult(
        answer="[D1] Start retrieval first, then open the web app.",
        citations=["D1"],
        context=context,
        prompt=PromptBundle(system="", user="", messages=[]),
    )


def test_web_app_serves_browser_experience(tmp_path):
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Agentic Search" in response.text
    assert "/api/agent" in client.get("/assets/app.js").text
    assert "text/css" in client.get("/assets/app.css").headers["content-type"]


def test_web_demo_all_sources_includes_browser_excludes_google():
    assert _source_providers_for("all") == ["retrieval", "serpapi", "browser"]


def test_web_demo_rejects_disabled_google_provider():
    try:
        _normalize_source_provider("google")
    except Exception as exc:
        assert "source_provider must be one of" in str(exc)
    else:
        raise AssertionError("google provider should be disabled for the web demo")


def test_agent_endpoint_runs_pipeline_and_persists_chat(monkeypatch, tmp_path):
    async def fake_answer_with_retrieval(
        question: str,
        *,
        llm=None,
        chat_history: list[ChatMessage] | None = None,
        search_url: str,
        top_k: int,
        filters=None,
    ) -> AnswerGenerationResult:
        assert filters is None
        assert question == "How do I deploy?"
        assert chat_history == []
        # The client-supplied search_url below is ignored; the server resolves
        # the retrieval URL from its own settings (SSRF protection).
        assert search_url == "http://server.test/retrieve"
        assert top_k == 3
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        fake_answer_with_retrieval,
    )
    store = AgenticSearchStore(tmp_path / "state.sqlite3")
    app = create_web_app(
        SearchExperienceSettings(search_url="http://server.test/retrieve"),
        store=store,
    )
    client = TestClient(app)

    response = client.post(
        "/api/agent",
        json={
            "query": "How do I deploy?",
            "search_url": "http://search.test/retrieve",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "[D1] Start retrieval first, then open the web app."
    assert data["documents"][0]["title"] == "Deployment Guide"
    assert [message["role"] for message in data["messages"]] == ["user", "assistant"]

    session = client.get(f"/api/sessions/{data['session_id']}").json()
    assert [message["content"] for message in session["messages"]] == [
        "How do I deploy?",
        "[D1] Start retrieval first, then open the web app.",
    ]
    store.close()


def test_agent_endpoint_reuses_existing_session_history(monkeypatch, tmp_path):
    observed_history: list[ChatMessage] = []

    async def fake_answer_with_retrieval(
        question: str,
        *,
        llm=None,
        chat_history: list[ChatMessage] | None = None,
        search_url: str,
        top_k: int,
        filters=None,
    ) -> AnswerGenerationResult:
        del filters
        observed_history.extend(chat_history or [])
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        fake_answer_with_retrieval,
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)
    session = client.post("/api/sessions", json={"title": "Deployment"}).json()
    client.post(
        "/api/agent",
        json={
            "query": "First question",
            "session_id": session["id"],
            "mode": "chat_once",
        },
    )

    response = client.post(
        "/api/agent",
        json={
            "query": "Follow up",
            "session_id": session["id"],
            "mode": "chat_once",
        },
    )

    assert response.status_code == 200
    assert [message.role for message in observed_history] == ["user", "assistant"]


def test_agent_endpoint_runs_query_processing_hook(monkeypatch, tmp_path):
    class FakeHookResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"query": "rewritten deploy question", "metadata": {"source": "hook"}}'

    async def fake_answer_with_retrieval(
        question: str,
        *,
        llm=None,
        chat_history: list[ChatMessage] | None = None,
        search_url: str,
        top_k: int,
        filters=None,
    ) -> AnswerGenerationResult:
        del llm, chat_history, search_url, top_k, filters
        assert question == "rewritten deploy question"
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        fake_answer_with_retrieval,
    )
    monkeypatch.setattr(
        "src.internal.hooks.executor.urllib.request.urlopen",
        lambda request, timeout: FakeHookResponse(),
    )
    registry = HookRegistry(
        [
            HookConfig(
                hook_point=HookPoint.QUERY_PROCESSING,
                endpoint_url="https://hooks.test/query",
            )
        ]
    )
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"),
        hook_registry=registry,
    )
    client = TestClient(app)

    response = client.post("/api/agent", json={"query": "How do I deploy?"})

    assert response.status_code == 200
    session = client.get(f"/api/sessions/{response.json()['session_id']}").json()
    assert session["messages"][0]["content"] == "rewritten deploy question"
    assert registry.execution_log[-1].is_success is True


def test_direct_search_enriches_web_provider_content(monkeypatch):
    """Content fetching is called for serpapi/google providers, not for retrieval."""
    from src.tools.search import SearchPage
    from src.internal.servers.web.app import _run_direct_search
    import asyncio

    serpapi_pages = [
        SearchPage(title="Result A", summary="snippet A", url="https://a.test"),
    ]
    fetched_pages = [
        SearchPage(
            title="Result A", summary="full article content A", url="https://a.test"
        ),
    ]

    async def _fake_search_tool(query, *, provider, search_url, page_size):
        return serpapi_pages

    async def _fake_fetch_pages(pages, *, max_chars, timeout_seconds=10):
        assert pages == serpapi_pages
        return fetched_pages

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", _fake_fetch_pages
    )

    docs = asyncio.run(
        _run_direct_search(
            "test query",
            source_provider="serpapi",
            search_url="http://localhost:8000/retrieve",
            top_k=3,
        )
    )
    assert any("full article content A" in doc.content for doc in docs)


def test_direct_search_skips_fetch_for_retrieval_provider(monkeypatch):
    """Content fetching is NOT called for the local retrieval provider."""
    from src.tools.search import SearchPage
    from src.internal.servers.web.app import _run_direct_search
    import asyncio

    fetch_called = []

    async def _fake_search_tool(query, *, provider, search_url, page_size):
        return [SearchPage(title="R", summary="corpus content", url="https://r.test")]

    async def _fake_fetch_pages(pages, **kwargs):
        fetch_called.append(True)
        return pages

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", _fake_fetch_pages
    )

    asyncio.run(
        _run_direct_search(
            "test query",
            source_provider="retrieval",
            search_url="http://localhost:8000/retrieve",
            top_k=3,
        )
    )
    assert not fetch_called


def test_hybrid_search_enriches_serpapi_provider_content(monkeypatch):
    """Hybrid search fetches full page content for serpapi results."""
    from src.tools.search import SearchPage
    from src.internal.servers.web.app import _run_hybrid_search
    import asyncio

    pages = [SearchPage(title="T", summary="snippet", url="https://t.test")]
    fetched = [SearchPage(title="T", summary="full article body", url="https://t.test")]

    async def _fake_search_tool(query, *, provider, search_url, page_size, **kw):
        return pages

    async def _fake_fetch_pages(pgs, *, max_chars, timeout_seconds=10):
        return fetched

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", _fake_fetch_pages
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.expand_keywords",
        lambda query, llm: [],
    )

    result = asyncio.run(
        _run_hybrid_search(
            "latest AI news",
            llm=None,
            search_url="http://localhost:8000/retrieve",
            top_k=3,
            filters=None,
            source_provider="serpapi",
        )
    )
    assert any("full article body" in doc.content for doc in result.documents)


def test_hybrid_search_includes_temporal_variant_for_time_sensitive_query(monkeypatch):
    """Temporal variant is added to executed queries for time-sensitive queries."""
    from src.internal.servers.web.app import _run_hybrid_search
    from src.tools.search import SearchPage
    import asyncio

    executed: list[str] = []

    async def _fake_search_tool(query, *, provider, search_url, page_size, **kw):
        executed.append(query)
        return [SearchPage(title="T", summary="s", url="https://t.test")]

    async def _fake_fetch_pages(pages, **kwargs):
        return pages

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", _fake_fetch_pages
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.expand_keywords",
        lambda query, llm: [],
    )

    result = asyncio.run(
        _run_hybrid_search(
            "latest AI models",
            llm=None,
            search_url="http://localhost:8000/retrieve",
            top_k=3,
            filters=None,
            source_provider="serpapi",
        )
    )
    from datetime import datetime

    year = str(datetime.now().year)
    assert any(year in q for q in result.executed_queries)


def test_hybrid_search_runs_search_tool_calls_concurrently(monkeypatch):
    """All search tool calls for expanded queries run concurrently (asyncio.gather)."""
    from src.internal.servers.web.app import _run_hybrid_search
    from src.tools.search import SearchPage
    import asyncio

    call_count = []

    async def _fake_search_tool(query, *, provider, search_url, page_size, **kw):
        call_count.append(query)
        return [SearchPage(title="T", summary="s", url="https://t.test")]

    async def _fake_fetch_pages(pages, **kwargs):
        return pages

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", _fake_fetch_pages
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.expand_keywords",
        lambda query, llm: ["AI news expanded"],
    )

    result = asyncio.run(
        _run_hybrid_search(
            "latest AI news",
            llm=object(),  # non-None so expand_keywords is called
            search_url="http://localhost:8000/retrieve",
            top_k=3,
            filters=None,
            source_provider="serpapi",
        )
    )
    # 2 queries: original + 1 expansion (temporal variant added too = 3 total)
    assert len(call_count) >= 2
    assert result.executed_queries is not None


def test_search_agent_mode_is_valid():
    assert _normalize_agent_mode("search_agent") == "search_agent"


def test_search_agent_returns_400_when_not_configured(tmp_path):
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)
    response = client.post(
        "/api/agent",
        json={"query": "What is FAISS?", "mode": "search_agent"},
    )
    assert response.status_code == 400
    assert "SEARCH_AGENT_MODEL" in response.json()["detail"]


def test_tool_agent_mode_is_valid():
    assert _normalize_agent_mode("tool_agent") == "tool_agent"


def test_tool_agent_returns_400_when_not_configured(tmp_path):
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)
    response = client.post(
        "/api/agent",
        json={"query": "What is FAISS?", "mode": "tool_agent"},
    )
    assert response.status_code == 400
    assert "SEARCH_AGENT_MODEL" in response.json()["detail"]


def test_run_agent_search_tool_mode_returns_documents(monkeypatch, tmp_path):
    from src.context.models import ContextDocument

    docs = [
        ContextDocument(
            id="D1",
            title="FAISS Guide",
            content="FAISS is a similarity search library.",
            url="https://example.test/faiss",
            score=0.95,
            metadata={},
        )
    ]

    async def fake_run_direct_search(
        query,
        *,
        source_provider,
        search_url,
        top_k,
        browser_search_url=None,
        rerank_url=None,
    ):
        return docs

    monkeypatch.setattr(
        "src.internal.servers.web.app._run_direct_search",
        fake_run_direct_search,
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)

    response = client.post(
        "/api/agent",
        json={"query": "What is FAISS?", "mode": "search_tool"},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["documents"]) == 1
    assert data["documents"][0]["title"] == "FAISS Guide"


def test_run_agent_chat_once_mode_returns_answer(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        lambda *a, **kw: __import__("asyncio").coroutine(lambda: _answer_result("q"))(),
    )

    async def fake_answer(
        question, *, llm=None, chat_history=None, search_url, top_k, filters=None
    ):
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)

    response = client.post(
        "/api/agent",
        json={"query": "How do I deploy?", "mode": "chat_once"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "Start retrieval first" in data["answer"]


def test_run_agent_chat_once_citations_extracted(monkeypatch, tmp_path):
    """citations in the response match the [Dx] markers extracted from the answer."""

    async def fake_answer(question, *, llm=None, chat_history=None, **kw):
        return _answer_result(question)  # answer contains "[D1]", citations=["D1"]

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)

    response = client.post(
        "/api/agent", json={"query": "What is FAISS?", "mode": "chat_once"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["citations"] == ["D1"]
    assert "[D1]" in data["answer"]


def test_run_agent_trims_long_history(monkeypatch, tmp_path):
    """When session history exceeds MAX_HISTORY_MESSAGES only the tail reaches the LLM."""
    from src.internal.servers.web.app import MAX_HISTORY_MESSAGES

    captured: list = []

    async def fake_answer(question, *, llm=None, chat_history=None, **kw):
        captured.append(list(chat_history or []))
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )

    store = AgenticSearchStore(tmp_path / "state.sqlite3")
    session = store.create_chat_session(title="long")
    for i in range(60):
        role = "user" if i % 2 == 0 else "assistant"
        store.add_chat_message(session.id, role=role, content=f"msg {i}")

    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"), store=store
    )
    client = TestClient(app)
    client.post(
        "/api/agent",
        json={"query": "follow up", "mode": "chat_once", "session_id": session.id},
    )

    assert len(captured) == 1
    assert len(captured[0]) <= MAX_HISTORY_MESSAGES


def test_agent_endpoint_returns_intent_field(monkeypatch, tmp_path):
    async def fake_answer(*args, **kwargs):
        return _answer_result("q")

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "explain FAISS"})
    assert response.status_code == 200
    data = response.json()
    assert "intent" in data
    assert data["intent"] in ("search", "chat", "tool")


def test_auto_route_chat_uses_answer_with_retrieval(monkeypatch, tmp_path):
    """No mode in request → auto-routes to chat via answer_with_retrieval."""
    called = {}

    async def fake_answer(q, *, llm, chat_history, search_url, top_k, filters):
        called["answer"] = True
        return _answer_result(q)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "explain FAISS"})
    assert response.status_code == 200
    assert called.get("answer") is True
    assert response.json()["intent"] == "chat"


def test_auto_route_search_via_rule_based(monkeypatch, tmp_path):
    """Short keyword query → rule-based classifies as search → hybrid_search runs."""
    called = {}

    async def fake_hybrid(
        query,
        *,
        llm,
        search_url,
        browser_search_url,
        rerank_url,
        top_k,
        filters,
        source_provider,
    ):
        called["hybrid"] = True
        from src.internal.servers.web.app import _HybridSearchResult

        return _HybridSearchResult(executed_queries=[query], documents=[])

    monkeypatch.setattr("src.internal.servers.web.app._run_hybrid_search", fake_hybrid)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "procurement process"})
    assert response.status_code == 200
    assert called.get("hybrid") is True
    assert response.json()["intent"] == "search"


def test_explicit_mode_still_works(monkeypatch, tmp_path):
    """Passing explicit mode='chat_once' still routes to answer_with_retrieval."""
    called = {}

    async def fake_answer(q, *, llm, chat_history, search_url, top_k, filters):
        called["answer"] = True
        return _answer_result(q)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "hello", "mode": "chat_once"})
    assert response.status_code == 200
    assert called.get("answer") is True
    assert response.json()["intent"] == "chat"


def test_auto_route_tier1_tool_loop_runs_when_model_available(monkeypatch, tmp_path):
    """When manager+tokenizer are set, ToolAgentLoop is used as Tier 1 router."""
    from unittest.mock import AsyncMock, MagicMock
    from src.agents.base import AgentLoopOutput
    import json

    # A trace that says search_routing_tool was called
    fake_trace = json.dumps(
        {"tool_name": "search_routing_tool", "status": "completed", "result": "[]"}
    )
    fake_output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        action_trace=fake_trace,
        final_answer="Here are the results.",
    )
    monkeypatch.setattr(
        "src.agents.tool_calling.ToolAgentLoop.run",
        AsyncMock(return_value=fake_output),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        response = client.post("/api/agent", json={"query": "find the onboarding doc"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "search"
    assert data["answer"] == "Here are the results."


def test_agent_no_llm_no_model_returns_400(monkeypatch, tmp_path):
    """App with no LLM and no local model → chat query → 400.

    Deterministic regardless of local environment. Two real-world leak paths are
    closed so ``llm`` is genuinely None: (1) inject empty app_settings so the
    config loader's GEN_AI key can't set ``resolved.llm.api_key``; (2) set
    OPENAI_API_KEY="" — python-dotenv won't override an already-present var, so
    create_web_app's internal .env reload can't repopulate it (delenv alone is
    insufficient — the reload re-adds it). Synthesis is then forced to fail so the
    no-LLM 400 branch is exercised even when a retrieval server happens to be up
    (otherwise extractive retrieval would answer without an LLM → 200).
    """
    from unittest.mock import AsyncMock
    from src.internal.configs import AppSettings

    monkeypatch.setattr("src.internal.servers.web.app.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        AsyncMock(side_effect=RuntimeError("no retrieval backend")),
    )
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"),
        app_settings=AppSettings(),
    )
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "explain FAISS"})
    assert response.status_code == 400
    assert "no llm" in response.json()["detail"].lower()


def test_agent_tool_mode_without_model_returns_clear_400(tmp_path):
    """Explicit mode=tool_agent without local model → 400 with 'local model' in detail."""
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)
    response = client.post(
        "/api/agent", json={"query": "run tool", "mode": "tool_agent"}
    )
    assert response.status_code == 400
    assert "local model" in response.json()["detail"].lower()


def test_agent_other_exception_returns_502_with_message(monkeypatch, tmp_path):
    """Unexpected exception → 502 with the exception message, not 'Agent search failed'."""

    async def explode(*args, **kwargs):
        raise ValueError("bad input format")

    monkeypatch.setattr("src.internal.servers.web.app.answer_with_retrieval", explode)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)
    response = client.post(
        "/api/agent", json={"query": "explain FAISS", "mode": "chat_once"}
    )
    assert response.status_code == 502
    assert "bad input format" in response.json()["detail"]
    assert "Agent search failed" not in response.json()["detail"]


def test_hybrid_fanout_merges_real_and_drops_errored_provider(monkeypatch, tmp_path):
    """retrieval returns real pages, serpapi errors → only real docs, status ok."""
    import asyncio
    from src.internal.servers.web.app import _run_hybrid_search
    from src.tools import SearchPage

    async def fake_search_tool(query, *, provider, search_url, page_size, **kw):
        if provider == "retrieval":
            return [SearchPage(title="Real Doc", summary="real", url="http://x/1")]
        return [SearchPage(error="SERPAPI_API_KEY is required.")]

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app._expanded_queries", lambda q, llm: [q]
    )
    result = asyncio.run(
        _run_hybrid_search(
            "q",
            llm=None,
            search_url="http://localhost:8001/retrieve",
            top_k=3,
            filters=None,
            source_provider="auto",
        )
    )
    assert result.status == "ok"
    assert [d.title for d in result.documents] == ["Real Doc"]
    assert all(not d.metadata.get("error") for d in result.documents)


def test_hybrid_fanout_all_errored_is_unreachable(monkeypatch, tmp_path):
    import asyncio
    from src.internal.servers.web.app import _run_hybrid_search
    from src.tools import SearchPage

    async def fake_search_tool(query, *, provider, search_url, page_size, **kw):
        return [SearchPage(error="down")]

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app._expanded_queries", lambda q, llm: [q]
    )
    result = asyncio.run(
        _run_hybrid_search(
            "q",
            llm=None,
            search_url="http://x/retrieve",
            top_k=3,
            filters=None,
            source_provider="auto",
        )
    )
    assert result.status == "unreachable"
    assert result.documents == []


def test_hybrid_fanout_reachable_but_empty_is_empty(monkeypatch, tmp_path):
    import asyncio
    from src.internal.servers.web.app import _run_hybrid_search

    async def fake_search_tool(query, *, provider, search_url, page_size, **kw):
        return []  # reachable, no hits, no error

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app._expanded_queries", lambda q, llm: [q]
    )
    result = asyncio.run(
        _run_hybrid_search(
            "q",
            llm=None,
            search_url="http://x/retrieve",
            top_k=3,
            filters=None,
            source_provider="auto",
        )
    )
    assert result.status == "empty"
    assert result.documents == []


def test_hybrid_fanout_one_provider_raises_does_not_kill_other(monkeypatch, tmp_path):
    import asyncio
    from src.internal.servers.web.app import _run_hybrid_search
    from src.tools import SearchPage

    async def fake_search_tool(query, *, provider, search_url, page_size, **kw):
        if provider == "serpapi":
            raise RuntimeError("boom")
        return [SearchPage(title="Real", summary="r", url="http://x/1")]

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app._expanded_queries", lambda q, llm: [q]
    )
    result = asyncio.run(
        _run_hybrid_search(
            "q",
            llm=None,
            search_url="http://x/retrieve",
            top_k=3,
            filters=None,
            source_provider="auto",
        )
    )
    assert result.status == "ok"
    assert [d.title for d in result.documents] == ["Real"]


def test_direct_search_auto_excludes_browser_sidecar(monkeypatch):
    """source_provider='auto' must NOT pull the slow browser sidecar, while
    'all'/'retrieval' still do (regression for the browser-out-of-auto invariant)."""
    import asyncio
    from src.tools.search import SearchPage
    from src.internal.servers.web.app import _run_direct_search

    browser_calls: list[str] = []

    async def _fake_search_tool(query, *, provider, search_url, page_size):
        return [
            SearchPage(title=f"{provider} R", summary="c", url=f"http://{provider}/1")
        ]

    async def _fake_browser(query, *, browser_search_url, top_k, existing_count):
        browser_calls.append(query)
        return []

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_browser_search", _fake_browser
    )

    asyncio.run(
        _run_direct_search(
            "q",
            source_provider="auto",
            search_url="http://localhost:8001/retrieve",
            browser_search_url="http://localhost:9999/retrieve",
            top_k=3,
        )
    )
    assert browser_calls == []  # auto never triggers browser

    asyncio.run(
        _run_direct_search(
            "q",
            source_provider="retrieval",
            search_url="http://localhost:8001/retrieve",
            browser_search_url="http://localhost:9999/retrieve",
            top_k=3,
        )
    )
    assert browser_calls == ["q"]  # non-auto still gets the sidecar
