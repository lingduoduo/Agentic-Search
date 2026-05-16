"""Reusable search helpers for function-calling tools."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Literal

from ..retrieval.context import SearchResult
from ..retrieval.client import SearchClient, SearchClientConfig, aiohttp
from .base import FunctionTool

SearchProvider = Literal["retrieval", "google", "serpapi"]

GOOGLE_SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
SERPAPI_SEARCH_ENDPOINT = "https://serpapi.com/search.json"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@dataclass(frozen=True)
class SearchPage:
    title: str = ""
    summary: str = ""
    url: str = ""
    error: str | None = None

    @classmethod
    def from_search_result(cls, result: SearchResult) -> "SearchPage":
        return cls(
            title=result.title or "",
            summary=_compact_contents(result.contents),
            url=result.url or "",
        )


async def google_custom_search(
    query: str,
    *,
    page: int = 1,
    page_size: int = 5,
    api_key: str | None = None,
    cse_id: str | None = None,
    timeout_seconds: int = 15,
) -> list[SearchPage]:
    """Search Google Custom Search and return normalized pages."""

    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    cse_id = cse_id or os.getenv("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        return [SearchPage(error="GOOGLE_API_KEY and GOOGLE_CSE_ID are required.")]

    try:
        data = await _get_json(
            GOOGLE_SEARCH_ENDPOINT,
            params={
                "key": api_key,
                "cx": cse_id,
                "q": query,
                "num": page_size,
                "start": (page - 1) * page_size + 1,
            },
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        return [SearchPage(error=str(exc))]
    return [
        SearchPage(
            title=item.get("title", ""),
            summary=item.get("snippet", ""),
            url=item.get("link", ""),
        )
        for item in data.get("items", [])
    ]


async def serpapi_search(
    query: str,
    *,
    page: int = 1,
    page_size: int = 5,
    api_key: str | None = None,
    timeout_seconds: int = 15,
) -> list[SearchPage]:
    """Search SerpAPI Google results and return normalized pages."""

    api_key = api_key or os.getenv("SERPAPI_API_KEY") or os.getenv("SERP_API_KEY")
    if not api_key:
        return [SearchPage(error="SERPAPI_API_KEY or SERP_API_KEY is required.")]

    try:
        data = await _get_json(
            SERPAPI_SEARCH_ENDPOINT,
            params={
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "num": page_size,
                "start": (page - 1) * page_size,
            },
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        return [SearchPage(error=str(exc))]

    pages = [
        SearchPage(
            title=item.get("title", ""),
            summary=item.get("snippet", ""),
            url=item.get("link", ""),
        )
        for item in data.get("organic_results", [])
    ]

    answer_box = data.get("answer_box", {})
    answer = answer_box.get("snippet") or answer_box.get("answer")
    if answer:
        pages.insert(
            0,
            SearchPage(
                title=answer_box.get("title", ""),
                summary=answer,
                url=answer_box.get("link", ""),
            ),
        )
    return pages[:page_size]


async def retrieval_search(
    query: str,
    *,
    search_url: str,
    page_size: int = 5,
    timeout_seconds: int = 10,
    max_retries: int = 3,
    fetch_url: str | None = None,
) -> list[SearchPage]:
    """Search the repo's /retrieve server and return normalized pages."""

    client = SearchClient(
        SearchClientConfig(
            url=search_url,
            topk=page_size,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            fetch_url=fetch_url,
        )
    )
    try:
        return [
            SearchPage.from_search_result(result)
            for result in await client.retrieve_one(query, topk=page_size)
        ]
    except Exception as exc:
        return [SearchPage(error=str(exc))]
    finally:
        await client.aclose()


async def search_tool(
    query: str,
    *,
    provider: SearchProvider = "retrieval",
    page: int = 1,
    page_size: int = 5,
    search_url: str = "http://localhost:8000/retrieve",
    timeout_seconds: int = 15,
    max_retries: int = 3,
    fetch_url: str | None = None,
) -> list[SearchPage]:
    """Route one query to a configured provider."""

    if provider == "retrieval":
        return await retrieval_search(
            query,
            search_url=search_url,
            page_size=page_size,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            fetch_url=fetch_url,
        )
    if provider == "google":
        return await google_custom_search(
            query,
            page=page,
            page_size=page_size,
            timeout_seconds=timeout_seconds,
        )
    if provider == "serpapi":
        return await serpapi_search(
            query,
            page=page,
            page_size=page_size,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError("provider must be 'retrieval', 'google', or 'serpapi'")


async def search_for_list(
    query: str,
    *,
    provider: SearchProvider = "retrieval",
    search_url: str = "http://localhost:8000/retrieve",
    page: int = 1,
    page_size: int = 5,
) -> list[dict[str, str]]:
    """Return normalized search results as dictionaries."""

    pages = await search_tool(
        query,
        provider=provider,
        search_url=search_url,
        page=page,
        page_size=page_size,
    )
    return [
        {
            "title": page.title,
            "summary": page.summary,
            "url": page.url,
            **({"error": page.error} if page.error else {}),
        }
        for page in pages
    ]


async def search_for_tool_string(
    query: str,
    *,
    provider: SearchProvider = "retrieval",
    search_url: str = "http://localhost:8000/retrieve",
    page: int = 1,
    page_size: int = 5,
) -> str:
    """Return search results formatted for a text-only tool response."""

    pages = await search_tool(
        query,
        provider=provider,
        search_url=search_url,
        page=page,
        page_size=page_size,
    )
    return format_search_pages(pages)


async def fetch_url(
    url: str, *, max_length: int = 500, timeout_seconds: int = 15
) -> str:
    """Fetch readable webpage text with lightweight HTML extraction."""

    if not url:
        return ""
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        headers = {"User-Agent": DEFAULT_USER_AGENT}
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                html = await response.text()
        return _html_to_text(html)[:max_length]
    except Exception as exc:
        return f"[fetch error] {exc}"


async def search_for_detail(
    query: str,
    *,
    provider: SearchProvider = "retrieval",
    search_url: str = "http://localhost:8000/retrieve",
    page: int = 1,
    page_size: int = 5,
    chunk_size: int = 500,
) -> str:
    """Search and fetch detailed content for each result URL."""

    pages = await search_tool(
        query,
        provider=provider,
        search_url=search_url,
        page=page,
        page_size=page_size,
    )
    contents = await asyncio.gather(
        *[
            fetch_url(page.url, max_length=chunk_size)
            for page in pages
            if not page.error
        ]
    )
    content_iter = iter(contents)

    sections: list[str] = []
    for page in pages:
        if page.error:
            sections.append(f"Error: {page.error}")
            continue
        sections.append(
            f"Title: {page.title}\nURL: {page.url}\nContent: {next(content_iter, '')}"
        )
    return "\n\n".join(sections) if sections else "No results found."


def build_search_tool(
    *,
    provider: SearchProvider = "retrieval",
    search_url: str = "http://localhost:8000/retrieve",
    page_size: int = 5,
) -> FunctionTool:
    """Build a FunctionTool for ToolAgentLoop search usage."""

    async def search(query: str) -> str:
        return await search_for_tool_string(
            query,
            provider=provider,
            search_url=search_url,
            page_size=page_size,
        )

    return FunctionTool(
        fn=search,
        name="search",
        description="Search for information on a topic.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    )


def format_search_pages(pages: list[SearchPage]) -> str:
    sections: list[str] = []
    for page in pages:
        if page.error:
            sections.append(f"Error: {page.error}")
            continue
        sections.append(
            f"Title: {page.title}\nSummary: {page.summary}\nURL: {page.url}"
        )
    return "\n\n".join(sections) if sections else "No results found."


async def _get_json(
    url: str,
    *,
    params: dict[str, Any],
    timeout_seconds: int,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, params=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()


def _compact_contents(contents: str, limit: int = 500) -> str:
    lines = [line.strip() for line in contents.splitlines() if line.strip()]
    if len(lines) > 1 and lines[0].startswith('"') and lines[0].endswith('"'):
        text = " ".join(lines[1:])
    else:
        text = " ".join(lines)
    return text[:limit]


def _html_to_text(html: str) -> str:
    try:
        import bs4
    except ImportError:
        return " ".join(html.split())

    soup = bs4.BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join(paragraph for paragraph in paragraphs if paragraph)
    return text or soup.get_text(" ", strip=True)
