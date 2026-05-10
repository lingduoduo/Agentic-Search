"""HTTP client for the repo's search and retrieval servers.

Servers may expose either:
    Request:  {"queries": ["..."], "topk": N}
    Response: {"result": [[item, ...], ...]}  — one inner list per query

or a trainer-friendly single-query shape:
    Request:  {"query": "...", "top_k": N}
    Response: {"query": "...", "results": [item, ...]}

Item shapes differ by server:
    retrieval_server (return_scores=True):  {"document": {...}, "score": float}
    retrieval_server (return_scores=False): document dict directly
    google_search_server / serp_search_server: {"document": {"contents": "..."}}

SearchResult.from_api_item handles all three shapes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from importlib import import_module
from urllib.parse import urlparse, urlunparse

from .context import SearchResult


class _LazyAiohttp:
    """Import aiohttp on first use while preserving monkeypatchable attributes."""

    def __getattr__(self, name: str):
        module = import_module("aiohttp")
        value = getattr(module, name)
        setattr(self, name, value)
        return value


aiohttp = _LazyAiohttp()


@dataclass(frozen=True)
class SearchClientConfig:
    url: str
    topk: int = 5
    timeout_seconds: int = 10
    max_retries: int = 3
    # Explicit /fetch endpoint. When None, derived from url by replacing /retrieve with /fetch.
    fetch_url: str | None = None

    def get_fetch_url(self) -> str:
        if self.fetch_url:
            return self.fetch_url
        parsed = urlparse(self.url)
        path = parsed.path.rstrip("/")
        if path.endswith("/retrieve"):
            path = path[: -len("/retrieve")]
        path = path.rstrip("/") + "/fetch"
        return urlunparse(parsed._replace(path=path, query="", fragment=""))


class SearchClient:
    """Async client for any POST /retrieve endpoint."""

    def __init__(self, config: SearchClientConfig) -> None:
        self.config = config
        self._timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _post_json(self, url: str, payload: dict, action: str) -> dict:
        last_exc: Exception | None = None

        for attempt in range(self.config.max_retries):
            session = await self._get_session()
            try:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as exc:
                if isinstance(exc, aiohttp.ClientResponseError) and exc.status < 500:
                    # 4xx errors are client-side mistakes — retrying won't help.
                    raise
                last_exc = exc
                if self._session is session:
                    await self.aclose()
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(0.5 * (2**attempt))

        raise RuntimeError(
            f"SearchClient.{action} failed after {self.config.max_retries} retries "
            f"against {url}"
        ) from last_exc

    async def retrieve(
        self,
        queries: list[str],
        topk: int | None = None,
    ) -> list[list[SearchResult]]:
        """Return one list of SearchResult per query.

        Raises RuntimeError after max_retries exhausted.
        Uses exponential backoff between retries.
        """
        payload = {"queries": queries, "topk": topk or self.config.topk}
        data = await self._post_json(self.config.url, payload, "retrieve")
        rows = data.get("result", data.get("results", []))
        if rows and isinstance(rows[0], dict):
            rows = [rows]
        return [[SearchResult.from_api_item(item) for item in row] for row in rows]

    async def retrieve_one(
        self,
        query: str,
        topk: int | None = None,
    ) -> list[SearchResult]:
        """Convenience wrapper for a single query."""
        results = await self.retrieve([query], topk=topk)
        return results[0] if results else []

    async def fetch_urls(
        self,
        urls: list[str],
    ) -> list[SearchResult]:
        """Fetch full-page content for specific URLs from a compatible server."""
        payload = {"urls": urls}
        fetch_url = self.config.get_fetch_url()
        data = await self._post_json(fetch_url, payload, "fetch_urls")
        return [SearchResult.from_api_item(item) for item in data.get("result", [])]
