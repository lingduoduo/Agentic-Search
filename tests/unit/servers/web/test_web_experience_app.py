from __future__ import annotations

from fastapi.testclient import TestClient

from src.context.models import AnswerGenerationResult
from src.context.models import ChatMessage
from src.context.models import ContextDocument
from src.context.models import PromptBundle
from src.context.models import SearchContextBundle
from src.backend.db import AgenticSearchStore
from src.backend.hooks import HookConfig
from src.backend.hooks import HookPoint
from src.backend.hooks import HookRegistry
from src.backend.servers.web.app import SearchExperienceSettings
from src.backend.servers.web.app import create_web_app
from src.backend.servers.web.app import _normalize_source_provider
from src.backend.servers.web.app import _source_providers_for


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


def test_web_demo_all_sources_excludes_disabled_google():
    assert _source_providers_for("all") == ["retrieval", "serpapi"]


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
        assert search_url == "http://search.test/retrieve"
        assert top_k == 3
        return _answer_result(question)

    monkeypatch.setattr(
        "src.backend.servers.web.app.answer_with_retrieval",
        fake_answer_with_retrieval,
    )
    store = AgenticSearchStore(tmp_path / "state.sqlite3")
    app = create_web_app(store=store)
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
        "src.backend.servers.web.app.answer_with_retrieval",
        fake_answer_with_retrieval,
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)
    session = client.post("/api/sessions", json={"title": "Deployment"}).json()
    client.post(
        "/api/agent",
        json={"query": "First question", "session_id": session["id"]},
    )

    response = client.post(
        "/api/agent",
        json={"query": "Follow up", "session_id": session["id"]},
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
        "src.backend.servers.web.app.answer_with_retrieval",
        fake_answer_with_retrieval,
    )
    monkeypatch.setattr(
        "src.backend.hooks.executor.urllib.request.urlopen",
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
    from src.backend.servers.web.app import _run_direct_search
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

    monkeypatch.setattr("src.backend.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr(
        "src.backend.servers.web.app.fetch_pages_concurrently", _fake_fetch_pages
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
    from src.backend.servers.web.app import _run_direct_search
    import asyncio

    fetch_called = []

    async def _fake_search_tool(query, *, provider, search_url, page_size):
        return [SearchPage(title="R", summary="corpus content", url="https://r.test")]

    async def _fake_fetch_pages(pages, **kwargs):
        fetch_called.append(True)
        return pages

    monkeypatch.setattr("src.backend.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr(
        "src.backend.servers.web.app.fetch_pages_concurrently", _fake_fetch_pages
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
    from src.backend.servers.web.app import _run_hybrid_search
    import asyncio

    pages = [SearchPage(title="T", summary="snippet", url="https://t.test")]
    fetched = [SearchPage(title="T", summary="full article body", url="https://t.test")]

    async def _fake_search_tool(query, *, provider, search_url, page_size):
        return pages

    async def _fake_fetch_pages(pgs, *, max_chars, timeout_seconds=10):
        return fetched

    monkeypatch.setattr("src.backend.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr(
        "src.backend.servers.web.app.fetch_pages_concurrently", _fake_fetch_pages
    )
    monkeypatch.setattr(
        "src.backend.servers.web.app.expand_keywords",
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
    from src.backend.servers.web.app import _run_hybrid_search
    from src.tools.search import SearchPage
    import asyncio

    executed: list[str] = []

    async def _fake_search_tool(query, *, provider, search_url, page_size):
        executed.append(query)
        return [SearchPage(title="T", summary="s", url="https://t.test")]

    async def _fake_fetch_pages(pages, **kwargs):
        return pages

    monkeypatch.setattr("src.backend.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr(
        "src.backend.servers.web.app.fetch_pages_concurrently", _fake_fetch_pages
    )
    monkeypatch.setattr(
        "src.backend.servers.web.app.expand_keywords",
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
