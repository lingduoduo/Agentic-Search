"""Search tools for the Agentic Search MCP server."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from src.tools.search import fetch_url
from src.tools.search import google_custom_search
from src.tools.search import retrieval_search
from src.tools.search import serper_dev_search
from src.tools.search import serpapi_search

from ..api import mcp_server

logger = logging.getLogger(__name__)


def _retrieval_url() -> str:
    port = os.getenv("AGENTIC_SEARCH_RETRIEVAL_PORT", "8000")
    return f"http://localhost:{port}/retrieve"


def _error_payload(error: str) -> dict[str, Any]:
    return {"error": error, "results": []}


@mcp_server.tool()
async def search_indexed_documents(
    query: str,
    source_types: list[str] | None = None,
    document_set_names: list[str] | None = None,
    time_cutoff: str | None = None,
    skip_query_expansion: bool = False,
) -> dict[str, Any]:
    """
    Search the knowledge base indexed in Agentic Search.
    Use this tool for information specific to your organization, team, or documents
    that have been indexed in the retrieval server.

    `source_types` and `document_set_names` are accepted for API compatibility but
    are not used for filtering — the retrieval server returns all indexed documents.
    `time_cutoff` and `skip_query_expansion` are similarly accepted but not applied.

    Returns ``{"results": [{title, url, content}, ...]}``.

    Example usage:
    ```
    {"query": "dense retrieval with FAISS"}
    ```
    """
    logger.info("MCP Server: document search: query='%s'", query)

    try:
        pages = await retrieval_search(query, search_url=_retrieval_url(), page_size=5)
    except Exception as err:
        logger.error("MCP Server: Document search error: %s", err, exc_info=True)
        return _error_payload(f"Document search failed: {str(err)}")

    if not pages:
        return _error_payload(
            "No results found. Ensure the retrieval server is running and documents are indexed."
        )

    results = [
        {"title": p.title, "url": p.url or "", "content": p.summary} for p in pages
    ]
    logger.info("MCP Server: document search returned %s results", len(results))
    return {"results": results}


@mcp_server.tool()
async def search_web(
    query: str,
    limit: int = 5,
) -> dict[str, Any]:
    """
    Search the public internet for general knowledge and current events.
    Use this tool for publicly available information such as news, documentation,
    or general facts.

    Returns ``{"results": [{title, url, snippet}, ...], "query": query}``.
    Use ``open_urls`` to fetch full content from returned URLs.

    The search provider is selected via the ``MCP_WEB_SEARCH_PROVIDER`` env var
    (``google``, ``serpapi``, or ``serper``; defaults to ``google``).

    Example usage:
    ```
    {"query": "React 19 release notes", "limit": 5}
    ```
    """
    logger.info("MCP Server: web search: query='%s', limit=%s", query, limit)

    provider = os.getenv("MCP_WEB_SEARCH_PROVIDER", "google")
    try:
        if provider == "serpapi":
            pages = await serpapi_search(query, page_size=limit)
        elif provider == "serper":
            pages = await serper_dev_search(query, page_size=limit)
        else:
            pages = await google_custom_search(query, page_size=limit)
    except Exception as exc:
        logger.error("MCP Server: Web search error: %s", exc, exc_info=True)
        return {
            "error": f"Web search failed: {str(exc)}",
            "results": [],
            "query": query,
        }

    results = [
        {"title": p.title, "url": p.url, "snippet": p.summary}
        for p in pages
        if not p.error
    ]
    errors = [p.error for p in pages if p.error]
    if errors:
        logger.warning("MCP Server: %d web search result(s) had errors", len(errors))

    return {"results": results, "query": query}


@mcp_server.tool()
async def open_urls(
    urls: list[str],
) -> dict[str, Any]:
    """
    Retrieve the complete text content from specific web URLs.
    Use this tool to fetch full page content for URLs returned by ``search_web``.

    Returns ``{"results": [{url, content}, ...]}``.

    Example usage:
    ```
    {"urls": ["https://react.dev/learn/react-compiler"]}
    ```
    """
    logger.info("MCP Server: open_urls: fetching %d URLs", len(urls))

    try:
        contents = await asyncio.gather(*[fetch_url(url) for url in urls])
    except Exception as err:
        logger.error("MCP Server: URL fetch error: %s", err, exc_info=True)
        return _error_payload(f"URL fetch failed: {str(err)}")

    results = [{"url": url, "content": content} for url, content in zip(urls, contents)]
    return {"results": results}
