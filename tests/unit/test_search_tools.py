"""Unit tests for function-calling search tools."""

from __future__ import annotations

import asyncio

from src.tools.search import (
    SearchPage,
    build_search_tool,
    format_search_pages,
    google_custom_search,
    search_for_detail,
    search_for_list,
    search_for_tool_string,
    serpapi_search,
)


class _FakeResponse:
    def __init__(self, payload=None, text=""):
        self._payload = payload or {}
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return self._payload

    async def text(self):
        return self._text


class _FakeSession:
    def __init__(self, *, payload=None, text="", calls=None, timeout=None):
        del timeout
        self._payload = payload or {}
        self._text = text
        self._calls = calls if calls is not None else []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.closed = True
        return False

    def get(self, url, **kwargs):
        self._calls.append((url, kwargs))
        return _FakeResponse(payload=self._payload, text=self._text)


def test_format_search_pages_handles_errors_and_empty_results():
    assert format_search_pages([]) == "No results found."
    assert (
        format_search_pages([SearchPage(title="T", summary="S", url="https://e.test")])
        == "Title: T\nSummary: S\nURL: https://e.test"
    )
    assert format_search_pages([SearchPage(error="boom")]) == "Error: boom"


def test_google_custom_search_maps_results_and_pagination(monkeypatch):
    calls = []

    def _session_factory(*, timeout):
        return _FakeSession(
            timeout=timeout,
            calls=calls,
            payload={
                "items": [
                    {"title": "One", "snippet": "Summary", "link": "https://one.test"}
                ]
            },
        )

    monkeypatch.setattr(
        "src.tools.search.aiohttp.ClientSession",
        _session_factory,
    )

    pages = asyncio.run(
        google_custom_search(
            "dense retrieval",
            page=2,
            page_size=3,
            api_key="key",
            cse_id="cx",
        )
    )

    assert pages == [SearchPage(title="One", summary="Summary", url="https://one.test")]
    assert calls[0][1]["params"]["start"] == 4
    assert calls[0][1]["params"]["num"] == 3


def test_serpapi_search_accepts_serp_api_key_env_alias(monkeypatch):
    calls = []
    monkeypatch.setenv("SERP_API_KEY", "serp-key")

    def _session_factory(*, timeout):
        return _FakeSession(
            timeout=timeout,
            calls=calls,
            payload={
                "answer_box": {"title": "Answer", "answer": "42", "link": "https://a"},
                "organic_results": [
                    {"title": "Organic", "snippet": "Body", "link": "https://o"}
                ],
            },
        )

    monkeypatch.setattr(
        "src.tools.search.aiohttp.ClientSession",
        _session_factory,
    )

    pages = asyncio.run(serpapi_search("answer", page_size=2))

    assert pages[0].title == "Answer"
    assert pages[1].title == "Organic"
    assert calls[0][1]["params"]["api_key"] == "serp-key"


def test_search_for_list_and_tool_string_use_retrieval_client(monkeypatch):
    async def _fake_retrieval_search(**kwargs):
        assert kwargs["query"] == "faiss"
        assert kwargs["page_size"] == 2
        return [SearchPage(title="FAISS", summary="Vector search", url="https://faiss")]

    monkeypatch.setattr(
        "src.tools.search.retrieval_search",
        lambda query, **kwargs: _fake_retrieval_search(query=query, **kwargs),
    )

    rows = asyncio.run(search_for_list("faiss", page_size=2))
    text = asyncio.run(search_for_tool_string("faiss", page_size=2))

    assert rows == [
        {"title": "FAISS", "summary": "Vector search", "url": "https://faiss"}
    ]
    assert "Title: FAISS" in text
    assert "Summary: Vector search" in text


def test_build_search_tool_wraps_formatted_search(monkeypatch):
    async def _fake_search_for_tool_string(query, **kwargs):
        assert query == "faiss"
        assert kwargs["page_size"] == 3
        return "formatted"

    monkeypatch.setattr(
        "src.tools.search.search_for_tool_string",
        _fake_search_for_tool_string,
    )

    tool = build_search_tool(page_size=3)
    text, raw, meta = asyncio.run(tool.execute("default", {"query": "faiss"}))

    assert text == "formatted"
    assert raw == "formatted"
    assert meta == {}


def test_search_for_detail_fetches_pages_concurrently(monkeypatch):
    async def _fake_search_tool(*args, **kwargs):
        del args, kwargs
        return [SearchPage(title="T", summary="S", url="https://t")]

    async def _fake_fetch_url(url, **kwargs):
        assert url == "https://t"
        assert kwargs["max_length"] == 20
        return "content"

    monkeypatch.setattr("src.tools.search.search_tool", _fake_search_tool)
    monkeypatch.setattr("src.tools.search.fetch_url", _fake_fetch_url)

    detail = asyncio.run(search_for_detail("query", chunk_size=20))

    assert detail == "Title: T\nURL: https://t\nContent: content"
