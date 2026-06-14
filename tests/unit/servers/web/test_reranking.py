"""Tests for browser hybrid merging and cross-encoder rerank integration."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.context.models import (
    AnswerGenerationResult,
    ContextDocument,
    PromptBundle,
    SearchContextBundle,
)
from src.tools.search import SearchPage
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


def _make_page(title: str, url: str, provider: str) -> SearchPage:
    return SearchPage(title=title, summary=f"summary from {provider}", url=url)


def _fake_answer(question: str) -> AnswerGenerationResult:
    doc = ContextDocument(
        id="D1", title="T", content="c", url="https://t.test", score=0.5
    )
    return AnswerGenerationResult(
        answer="ok",
        citations=["D1"],
        context=SearchContextBundle(query=question, documents=[doc]),
        prompt=PromptBundle(system="", user="", messages=[]),
    )


def test_browser_search_url_causes_browser_provider_call(tmp_path, monkeypatch):
    """When browser_search_url is set, search_tool is called with that URL."""
    call_log: list[tuple[str, str]] = []

    async def fake_search(query, *, provider, search_url, page_size=5):
        call_log.append((provider, search_url))
        return [_make_page("T", "https://t.test", provider)]

    async def fake_answer(question, *, context, llm_client=None):
        return _fake_answer(question)

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search)
    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )

    app = create_web_app(
        SearchExperienceSettings(
            db_path=tmp_path / "s.sqlite3",
            browser_search_url="http://browser.test:8002",
        )
    )
    client = TestClient(app)
    resp = client.post(
        "/api/agent",
        json={
            "query": "hybrid test",
            "mode": "search_tool",
            "source_provider": "retrieval",
        },
    )
    assert resp.status_code == 200
    browser_calls = [(p, u) for p, u in call_log if u == "http://browser.test:8002"]
    assert browser_calls, f"browser_search_url not used; calls were {call_log}"


def test_rerank_url_causes_rerank_documents_call(tmp_path, monkeypatch):
    """When rerank_url is set, _rerank_documents is called with that URL."""
    rerank_calls: list[str] = []

    async def fake_search(query, *, provider, search_url, page_size=5):
        return [_make_page("T", "https://t.test", provider)]

    async def fake_answer(question, *, context, llm_client=None):
        return _fake_answer(question)

    async def fake_rerank(docs, query, rerank_url):
        rerank_calls.append(rerank_url)
        return docs

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search)
    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )
    monkeypatch.setattr("src.internal.servers.web.app._rerank_documents", fake_rerank)

    app = create_web_app(
        SearchExperienceSettings(
            db_path=tmp_path / "s.sqlite3",
            rerank_url="http://rerank.test:6980",
        )
    )
    client = TestClient(app)
    resp = client.post(
        "/api/agent",
        json={"query": "rerank test", "mode": "search_tool"},
    )
    assert resp.status_code == 200
    assert rerank_calls == ["http://rerank.test:6980"]
