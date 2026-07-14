"""Tests for browser hybrid merging and cross-encoder rerank integration."""

from __future__ import annotations

import pytest
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

    async def fake_search(query, *, provider, search_url, page_size=5, **_):
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


@pytest.mark.asyncio
async def test_rerank_documents_returns_original_on_http_error(monkeypatch):
    """_rerank_documents falls back to original order when httpx raises."""
    import httpx
    from src.internal.servers.web.app import _rerank_documents
    from src.context.models import ContextDocument

    async def bad_post(self, url, *, json=None, timeout=None):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("httpx.AsyncClient.post", bad_post)

    docs = [
        ContextDocument(id="D1", title="A", content="a", score=0.5),
        ContextDocument(id="D2", title="B", content="b", score=0.3),
    ]
    result = await _rerank_documents(docs, "query", "http://rerank.test:6980")
    assert [doc.title for doc in result] == ["A", "B"]
    assert [doc.score for doc in result] == [0.5, 0.3]


@pytest.mark.asyncio
async def test_rerank_documents_updates_scores_and_reorders(monkeypatch):
    """_rerank_documents reorders docs by cross-encoder score."""
    import httpx
    from src.internal.servers.web.app import _rerank_documents
    from src.context.models import ContextDocument

    async def fake_post(self, url, *, json=None, timeout=None):
        # Simulate reranker reversing the order: D2 (idx=1) scores higher
        body = {
            "result": [
                [
                    {"document": {"contents": "B\nb", "_idx": "1"}, "score": 0.9},
                    {"document": {"contents": "A\na", "_idx": "0"}, "score": 0.4},
                ]
            ]
        }
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)

    docs = [
        ContextDocument(id="D1", title="A", content="a", score=0.0),
        ContextDocument(id="D2", title="B", content="b", score=0.0),
    ]
    result = await _rerank_documents(docs, "query", "http://rerank.test:6980")
    assert len(result) == 2
    assert result[0].title == "B"  # D2 now first (score 0.9)
    assert result[0].score == pytest.approx(0.9)
    assert result[1].title == "A"
    assert result[1].score == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_rerank_documents_drops_items_with_missing_idx(monkeypatch):
    """_rerank_documents silently drops items where _idx is absent or invalid."""
    import httpx
    from src.internal.servers.web.app import _rerank_documents
    from src.context.models import ContextDocument

    async def fake_post(self, url, *, json=None, timeout=None):
        body = {
            "result": [
                [
                    {
                        "document": {"contents": "no idx here"},
                        "score": 0.8,
                    },  # missing _idx
                    {"document": {"contents": "A\na", "_idx": "0"}, "score": 0.5},
                ]
            ]
        }
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)

    docs = [ContextDocument(id="D1", title="A", content="a", score=0.0)]
    result = await _rerank_documents(docs, "query", "http://rerank.test:6980")
    # The item with no _idx is dropped; the valid item is returned
    assert len(result) == 1
    assert result[0].title == "A"
    assert result[0].score == pytest.approx(0.5)


def test_rerank_url_is_forwarded_to_shared_ranking_stage(tmp_path, monkeypatch):
    """The web path delegates its configured reranker to shared ranking."""
    rerank_calls: list[str] = []

    async def fake_search(query, *, provider, search_url, page_size=5, **_):
        return [_make_page("T", "https://t.test", provider)]

    async def fake_answer(question, *, context, llm_client=None):
        return _fake_answer(question)

    async def fake_rank(docs, query, rerank_url, top_k):
        rerank_calls.append(rerank_url)
        return docs

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search)
    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )
    monkeypatch.setattr("src.internal.servers.web.app._rank_documents", fake_rank)

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
