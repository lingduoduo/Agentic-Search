"""Unit tests for the MCP server tools, resources, and utils."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from src.tools.search import SearchPage


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------


def test_build_web_base_url_defaults():
    from src.internal.mcp_server.utils import build_web_base_url

    with patch.dict(os.environ, {}, clear=False):
        for var in (
            "API_SERVER_URL_OVERRIDE_FOR_HTTP_REQUESTS",
            "API_SERVER_PROTOCOL",
            "API_SERVER_HOST",
            "AGENTIC_SEARCH_WEB_PORT",
        ):
            os.environ.pop(var, None)
        url = build_web_base_url()
    assert url == "http://127.0.0.1:7860"


def test_build_web_base_url_override():
    from src.internal.mcp_server.utils import build_web_base_url

    with patch.dict(
        os.environ,
        {"API_SERVER_URL_OVERRIDE_FOR_HTTP_REQUESTS": "https://my.server/"},
    ):
        url = build_web_base_url()
    assert url == "https://my.server"  # trailing slash stripped


def test_build_web_base_url_custom_port():
    from src.internal.mcp_server.utils import build_web_base_url

    with patch.dict(os.environ, {"AGENTIC_SEARCH_WEB_PORT": "9999"}, clear=False):
        os.environ.pop("API_SERVER_URL_OVERRIDE_FOR_HTTP_REQUESTS", None)
        url = build_web_base_url()
    assert url.endswith(":9999")


# ---------------------------------------------------------------------------
# api helpers
# ---------------------------------------------------------------------------


def test_get_cors_origins_empty():
    from src.internal.mcp_server.api import _get_cors_origins

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MCP_SERVER_CORS_ORIGINS", None)
        assert _get_cors_origins() == []


def test_get_cors_origins_multiple():
    from src.internal.mcp_server.api import _get_cors_origins

    with patch.dict(
        os.environ,
        {"MCP_SERVER_CORS_ORIGINS": "https://a.com, https://b.com"},
    ):
        result = _get_cors_origins()
    assert result == ["https://a.com", "https://b.com"]


# ---------------------------------------------------------------------------
# tools/search — search_indexed_documents
# ---------------------------------------------------------------------------


_FAKE_PAGES = [
    SearchPage(
        title="Dense Retrieval",
        summary="FAISS-based dense retrieval.",
        url="http://ex.com/1",
    ),
    SearchPage(
        title="Sparse BM25",
        summary="BM25 scoring for keyword search.",
        url="http://ex.com/2",
    ),
]


@pytest.mark.asyncio
async def test_search_indexed_documents_returns_results():
    from src.internal.mcp_server.tools.search import search_indexed_documents

    with patch(
        "src.internal.mcp_server.tools.search.retrieval_search",
        new=AsyncMock(return_value=_FAKE_PAGES),
    ):
        result = await search_indexed_documents(query="dense retrieval")

    assert "results" in result
    assert len(result["results"]) == 2
    assert result["results"][0]["title"] == "Dense Retrieval"
    assert result["results"][0]["url"] == "http://ex.com/1"
    assert "error" not in result


@pytest.mark.asyncio
async def test_search_indexed_documents_empty():
    from src.internal.mcp_server.tools.search import search_indexed_documents

    with patch(
        "src.internal.mcp_server.tools.search.retrieval_search",
        new=AsyncMock(return_value=[]),
    ):
        result = await search_indexed_documents(query="nothing")

    assert "error" in result
    assert result["results"] == []


@pytest.mark.asyncio
async def test_search_indexed_documents_error():
    from src.internal.mcp_server.tools.search import search_indexed_documents

    with patch(
        "src.internal.mcp_server.tools.search.retrieval_search",
        new=AsyncMock(side_effect=ConnectionError("retrieval server down")),
    ):
        result = await search_indexed_documents(query="test")

    assert "error" in result
    assert "retrieval server down" in result["error"]


# ---------------------------------------------------------------------------
# tools/search — search_web
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_web_google_provider():
    from src.internal.mcp_server.tools.search import search_web

    with patch.dict(os.environ, {"MCP_WEB_SEARCH_PROVIDER": "google"}):
        with patch(
            "src.internal.mcp_server.tools.search.google_custom_search",
            new=AsyncMock(return_value=_FAKE_PAGES),
        ):
            result = await search_web(query="FAISS", limit=2)

    assert result["query"] == "FAISS"
    assert len(result["results"]) == 2
    assert "error" not in result


@pytest.mark.asyncio
async def test_search_web_serpapi_provider():
    from src.internal.mcp_server.tools.search import search_web

    with patch.dict(os.environ, {"MCP_WEB_SEARCH_PROVIDER": "serpapi"}):
        with patch(
            "src.internal.mcp_server.tools.search.serpapi_search",
            new=AsyncMock(return_value=_FAKE_PAGES),
        ):
            result = await search_web(query="BM25")

    assert len(result["results"]) == 2


@pytest.mark.asyncio
async def test_search_web_filters_error_pages():
    from src.internal.mcp_server.tools.search import search_web

    pages_with_error = [
        SearchPage(title="OK", summary="good result", url="http://ex.com/ok"),
        SearchPage(error="rate limited"),
    ]
    with patch.dict(os.environ, {"MCP_WEB_SEARCH_PROVIDER": "google"}):
        with patch(
            "src.internal.mcp_server.tools.search.google_custom_search",
            new=AsyncMock(return_value=pages_with_error),
        ):
            result = await search_web(query="test")

    # Error pages are filtered from results
    assert len(result["results"]) == 1
    assert result["results"][0]["title"] == "OK"


@pytest.mark.asyncio
async def test_search_web_provider_exception():
    from src.internal.mcp_server.tools.search import search_web

    with patch.dict(os.environ, {"MCP_WEB_SEARCH_PROVIDER": "google"}):
        with patch(
            "src.internal.mcp_server.tools.search.google_custom_search",
            new=AsyncMock(side_effect=RuntimeError("network error")),
        ):
            result = await search_web(query="test")

    assert "error" in result
    assert result["results"] == []


# ---------------------------------------------------------------------------
# tools/search — open_urls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_urls_fetches_each():
    from src.internal.mcp_server.tools.search import open_urls

    async def _fake_fetch(url: str, **kwargs: object) -> str:
        return f"content of {url}"

    with patch(
        "src.internal.mcp_server.tools.search.fetch_url", side_effect=_fake_fetch
    ):
        result = await open_urls(urls=["http://a.com", "http://b.com"])

    assert len(result["results"]) == 2
    assert result["results"][0] == {
        "url": "http://a.com",
        "content": "content of http://a.com",
    }


@pytest.mark.asyncio
async def test_open_urls_error():
    from src.internal.mcp_server.tools.search import open_urls

    with patch(
        "src.internal.mcp_server.tools.search.fetch_url",
        side_effect=RuntimeError("timeout"),
    ):
        result = await open_urls(urls=["http://ex.com"])

    assert "error" in result


# ---------------------------------------------------------------------------
# resources
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_indexed_sources_retrieval_only():
    from src.internal.mcp_server.resources.indexed_sources import (
        indexed_sources_resource,
    )

    env = {
        k: ""
        for k in (
            "GOOGLE_API_KEY",
            "GOOGLE_CSE_ID",
            "SERPAPI_API_KEY",
            "SERP_API_KEY",
            "SERPER_API_KEY",
        )
    }
    with patch.dict(os.environ, env):
        result = json.loads(await indexed_sources_resource())
    assert result == ["retrieval"]


@pytest.mark.asyncio
async def test_indexed_sources_with_google():
    from src.internal.mcp_server.resources.indexed_sources import (
        indexed_sources_resource,
    )

    with patch.dict(
        os.environ,
        {
            "GOOGLE_API_KEY": "key",
            "GOOGLE_CSE_ID": "cse",
            "SERPER_API_KEY": "",
            "SERPAPI_API_KEY": "",
            "SERP_API_KEY": "",
        },
    ):
        result = json.loads(await indexed_sources_resource())
    assert "google" in result
    assert "retrieval" in result


@pytest.mark.asyncio
async def test_document_sets_returns_empty():
    from src.internal.mcp_server.resources.document_sets import document_sets_resource

    result = json.loads(await document_sets_resource())
    assert result == []
