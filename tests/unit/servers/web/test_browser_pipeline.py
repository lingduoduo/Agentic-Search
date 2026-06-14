"""Tests for the browser-search → retriever pipeline in app.py.

Covers three gaps not tested elsewhere:
  1. _run_browser_search — calls search_tool with provider="retrieval" and
     browser_search_url, converts results to ContextDocuments with source="browser".
  2. Browser docs bypass fetch_pages_concurrently (_is_web_provider("browser") is False).
  3. Browser sidecar merge: browser docs are merged with primary docs before
     dedup / rerank / MMR, and mmr_rank is stamped on every result.
"""

from __future__ import annotations


import pytest

from src.context.models import ContextDocument
from src.internal.servers.web.app import (
    _is_web_provider,
    _reindex_documents,
    _run_browser_search,
    _run_direct_search,
)
from src.tools.search import SearchPage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _page(title: str, url: str, summary: str = "text") -> SearchPage:
    return SearchPage(title=title, summary=summary, url=url)


# ---------------------------------------------------------------------------
# 1. _run_browser_search unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_browser_search_returns_context_documents(monkeypatch):
    """_run_browser_search converts SearchPages to ContextDocuments with source=browser."""
    pages = [
        _page("FAISS overview", "https://faiss.ai"),
        _page("GitHub FAISS", "https://github.com/facebookresearch/faiss"),
    ]

    async def fake_search(query, *, provider, search_url, page_size):
        assert provider == "retrieval"
        assert search_url == "http://browser.test:8002"
        return pages

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search)

    docs = await _run_browser_search(
        "what is FAISS",
        browser_search_url="http://browser.test:8002",
        top_k=5,
        existing_count=0,
    )

    assert len(docs) == 2
    assert all(isinstance(d, ContextDocument) for d in docs)
    assert all(d.metadata["source_provider"] == "browser" for d in docs)
    assert all(d.metadata["source"] == "Browser Retrieval" for d in docs)
    assert docs[0].title == "FAISS overview"
    assert docs[0].url == "https://faiss.ai"


@pytest.mark.asyncio
async def test_run_browser_search_ids_start_after_existing_count(monkeypatch):
    """Document IDs offset by existing_count so they don't collide with primary docs."""

    async def fake_search(query, *, provider, search_url, page_size):
        return [_page("T", "https://t.test")]

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search)

    docs = await _run_browser_search(
        "query",
        browser_search_url="http://browser.test:8002",
        top_k=5,
        existing_count=3,
    )

    assert docs[0].id == "D4"


@pytest.mark.asyncio
async def test_run_browser_search_returns_empty_on_exception(monkeypatch):
    """_run_browser_search returns [] and does not raise when search_tool fails."""

    async def failing_search(query, *, provider, search_url, page_size):
        raise ConnectionRefusedError("browser server not running")

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", failing_search)

    docs = await _run_browser_search(
        "query",
        browser_search_url="http://browser.test:8002",
        top_k=5,
        existing_count=0,
    )

    assert docs == []


# ---------------------------------------------------------------------------
# 2. _is_web_provider — browser bypasses fetch_pages_concurrently
# ---------------------------------------------------------------------------


def test_is_web_provider_returns_false_for_browser():
    """`browser` provider must NOT trigger fetch_pages_concurrently (already fetched)."""
    assert _is_web_provider("browser") is False


def test_is_web_provider_returns_true_for_serpapi():
    assert _is_web_provider("serpapi") is True


def test_is_web_provider_returns_false_for_google():
    # google uses search_tool internally; fetch_pages_concurrently not needed
    assert _is_web_provider("google") is False


def test_is_web_provider_returns_false_for_retrieval():
    assert _is_web_provider("retrieval") is False


@pytest.mark.asyncio
async def test_browser_docs_skip_fetch_pages_concurrently(monkeypatch):
    """When source_provider='browser', fetch_pages_concurrently is never called."""
    fetch_called: list[bool] = []

    async def fake_search(query, *, provider, search_url, page_size):
        return [_page("T", "https://t.test")]

    async def fake_fetch(pages, *, max_chars, timeout_seconds=10):
        fetch_called.append(True)
        return pages

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search)
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", fake_fetch
    )

    await _run_direct_search(
        "query",
        source_provider="browser",
        search_url="http://browser.test:8002",
        top_k=3,
    )

    assert not fetch_called, (
        "fetch_pages_concurrently must not be called for browser provider"
    )


# ---------------------------------------------------------------------------
# 3. Browser sidecar merge → dedup → MMR → mmr_rank stamped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_direct_search_merges_browser_sidecar_before_mmr(monkeypatch):
    """Browser sidecar docs are included in the MMR candidate pool."""
    primary_page = _page("Primary result", "https://primary.test", "primary content")
    browser_page = _page(
        "Browser result", "https://browser-result.test", "browser content"
    )

    async def fake_search(query, *, provider, search_url, page_size):
        if search_url == "http://browser.test:8002":
            return [browser_page]
        return [primary_page]

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search)

    # source_provider="retrieval" + browser_search_url → sidecar adds browser docs
    docs = await _run_direct_search(
        "query",
        source_provider="retrieval",
        search_url="http://retrieval.test:8001",
        browser_search_url="http://browser.test:8002",
        top_k=5,
    )

    sources = {d.metadata.get("source_provider") for d in docs}
    assert "browser" in sources, "browser sidecar docs must appear in merged output"
    assert "retrieval" in sources, "primary retrieval docs must also appear"


@pytest.mark.asyncio
async def test_direct_search_deduplicates_browser_and_primary_overlap(monkeypatch):
    """If browser and primary return the same URL, only one copy survives dedup."""
    shared_url = "https://shared.test"
    shared_page = _page("Shared doc", shared_url, "shared content here")

    async def fake_search(query, *, provider, search_url, page_size):
        return [shared_page]

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search)

    docs = await _run_direct_search(
        "query",
        source_provider="retrieval",
        search_url="http://retrieval.test:8001",
        browser_search_url="http://browser.test:8002",
        top_k=5,
    )

    urls = [d.url for d in docs]
    assert urls.count(shared_url) == 1, "duplicate URL must be deduplicated"


@pytest.mark.asyncio
async def test_reindex_documents_stamps_mmr_rank():
    """_reindex_documents adds mmr_rank=N (1-based) to every document's metadata."""
    docs = [
        ContextDocument(
            id="D1", title="A", content="a", score=0.9, metadata={"source": "x"}
        ),
        ContextDocument(
            id="D2", title="B", content="b", score=0.5, metadata={"source": "x"}
        ),
    ]
    reindexed = _reindex_documents(docs)

    assert reindexed[0].metadata["mmr_rank"] == 1
    assert reindexed[1].metadata["mmr_rank"] == 2
    assert reindexed[0].id == "D1"
    assert reindexed[1].id == "D2"


@pytest.mark.asyncio
async def test_browser_docs_have_mmr_rank_after_full_pipeline(monkeypatch):
    """Browser docs flowing through the full pipeline have mmr_rank stamped."""
    browser_page = _page(
        "Browser result", "https://browser-result.test", "browser content"
    )

    async def fake_search(query, *, provider, search_url, page_size):
        if search_url == "http://browser.test:8002":
            return [browser_page]
        return []

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search)

    docs = await _run_direct_search(
        "query",
        source_provider="browser",
        search_url="http://retrieval.test:8001",
        browser_search_url="http://browser.test:8002",
        top_k=5,
    )

    browser_docs = [d for d in docs if d.metadata.get("source_provider") == "browser"]
    assert browser_docs, "browser doc must survive the pipeline"
    for doc in browser_docs:
        assert "mmr_rank" in doc.metadata, f"mmr_rank missing from browser doc {doc.id}"
        assert isinstance(doc.metadata["mmr_rank"], int)
