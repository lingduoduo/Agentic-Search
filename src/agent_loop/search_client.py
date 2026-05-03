"""HTTP client for the repo's search and retrieval servers.

All three servers expose the same POST /retrieve interface:
    Request:  {"queries": ["..."], "topk": N}
    Response: {"result": [[item, ...], ...]}  — one inner list per query

Item shapes differ by server:
    retrieval_server (return_scores=True):  {"document": {...}, "score": float}
    retrieval_server (return_scores=False): document dict directly
    google_search_server / serp_search_server: {"document": {"contents": "..."}}

SearchResult.from_api_item handles all three shapes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import requests

from .context import SearchResult


@dataclass(frozen=True)
class SearchClientConfig:
    url: str
    topk: int = 5
    timeout_seconds: int = 10
    max_retries: int = 3


class SearchClient:
    """Thin synchronous client for any POST /retrieve endpoint."""

    def __init__(self, config: SearchClientConfig) -> None:
        self.config = config

    def retrieve(
        self,
        queries: list[str],
        topk: int | None = None,
    ) -> list[list[SearchResult]]:
        """Return one list of SearchResult per query.

        Raises RuntimeError after max_retries exhausted.
        """
        payload = {"queries": queries, "topk": topk or self.config.topk}
        last_exc: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                resp = requests.post(
                    self.config.url,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                resp.raise_for_status()
                rows = resp.json().get("result", [])
                return [
                    [SearchResult.from_api_item(item) for item in row]
                    for row in rows
                ]
            except Exception as exc:
                last_exc = exc
                if attempt < self.config.max_retries - 1:
                    time.sleep(0.5)

        raise RuntimeError(
            f"SearchClient.retrieve failed after {self.config.max_retries} retries "
            f"against {self.config.url}"
        ) from last_exc

    def retrieve_one(
        self,
        query: str,
        topk: int | None = None,
    ) -> list[SearchResult]:
        """Convenience wrapper for a single query."""
        results = self.retrieve([query], topk=topk)
        return results[0] if results else []
