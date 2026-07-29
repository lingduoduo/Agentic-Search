"""Tests for the auto-mode hybrid web cascade in app.py.

In 'auto' (internal + web) mode the web leg cascades: SerpAPI first, falling
back to the browser provider when SerpAPI yields no usable results (missing key,
error, or empty). Local retrieval runs in parallel; MMR picks the final top_k.
"""

from __future__ import annotations

import pytest

from src.context.models import ContextDocument
from src.internal.servers.web.app import _run_hybrid_search
from src.internal.tools.search import SearchPage


async def _passthrough_fetch(pages, **_kwargs):
    return pages


async def _run_auto(monkeypatch, *, serpapi_pages, browser_docs, browser_url):
    """Drive _run_hybrid_search in auto mode with local retrieval empty."""

    async def fake_search_tool(query, *, provider, **_kwargs):
        if provider == "retrieval":
            return []  # query absent from local corpus (e.g. GRPO)
        if provider == "serpapi":
            return serpapi_pages
        raise AssertionError(f"unexpected provider {provider}")

    async def fake_browser(query, *, browser_search_url, top_k, existing_count):
        if browser_docs is None:
            raise AssertionError("browser must not be called when SerpAPI succeeds")
        return browser_docs

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", _passthrough_fetch
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_browser_search", fake_browser
    )

    return await _run_hybrid_search(
        "GRPO",
        llm=None,
        search_url="http://retrieval",
        browser_search_url=browser_url,
        rerank_url=None,
        top_k=5,
        filters=None,
        source_provider="auto",
    )


@pytest.mark.asyncio
async def test_auto_falls_back_to_browser_when_serpapi_unusable(monkeypatch):
    """SerpAPI returns only an error page → browser fallback supplies results."""
    browser_doc = ContextDocument(
        id="D1",
        title="GRPO explained",
        content="Group Relative Policy Optimization is an RL method...",
        url="https://example.com/grpo",
        score=0.0,
        metadata={"error": False, "source_provider": "browser"},
    )
    result = await _run_auto(
        monkeypatch,
        serpapi_pages=[SearchPage(error="SERP_API_KEY is required.")],
        browser_docs=[browser_doc],
        browser_url="http://browser",
    )
    assert result.status == "ok"
    assert [d.title for d in result.documents] == ["GRPO explained"]


@pytest.mark.asyncio
async def test_auto_uses_serpapi_and_skips_browser_when_usable(monkeypatch):
    """SerpAPI returns real results → browser fallback is never consulted."""
    result = await _run_auto(
        monkeypatch,
        serpapi_pages=[SearchPage(title="GRPO paper", summary="...", url="https://a")],
        browser_docs=None,  # fake_browser raises if called
        browser_url="http://browser",
    )
    assert result.status == "ok"
    assert any("GRPO" in d.title for d in result.documents)


@pytest.mark.asyncio
async def test_auto_no_web_backend_returns_unreachable(monkeypatch):
    """No SerpAPI key and no browser URL → web leg errors, local empty → unreachable."""
    result = await _run_auto(
        monkeypatch,
        serpapi_pages=[SearchPage(error="SERP_API_KEY is required.")],
        browser_docs=[],
        browser_url=None,
    )
    assert result.status == "unreachable"
    assert result.documents == []
