"""Redis-backed cache for reranker scores.

Cache key: sha256(canonical_query + json(sorted_doc_ids))[:20].
A cache hit skips the base reranker entirely.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from typing import Any

from src.internal.retrieval.backends.base import RetrievalResult

logger = logging.getLogger(__name__)


def _cache_key(query: str, doc_ids: list[str]) -> str:
    canonical = query.lower().strip()
    sorted_ids = json.dumps(sorted(doc_ids))
    raw = f"{canonical}:{sorted_ids}"
    return "rrk:" + hashlib.sha256(raw.encode()).hexdigest()[:20]


class CachedReranker:
    """Redis-backed cache for reranker scores. Cache key: query + sorted doc_ids.

    Args:
        base_reranker: A reranker object with rerank(query, results, top_k) -> list[RetrievalResult].
        redis_client: A redis.Redis instance, or None to disable caching.
        ttl_seconds: Cache entry TTL (default 300s).
    """

    def __init__(
        self,
        base_reranker,
        redis_client: Any | None = None,
        *,
        ttl_seconds: int = 300,
    ) -> None:
        self._base = base_reranker
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        doc_ids = [r.doc_id for r in results]
        key = _cache_key(query, doc_ids)

        if self._redis is not None:
            raw = self._redis.get(key)
            if raw is not None:
                self._hits += 1
                return [RetrievalResult(**row) for row in json.loads(raw)]

        self._misses += 1
        reranked = self._base.rerank(query, results, top_k)

        if self._redis is not None:
            payload = json.dumps([dataclasses.asdict(r) for r in reranked])
            self._redis.setex(key, self._ttl, payload)

        return reranked

    def stats(self) -> dict:
        """Return cache statistics: hits, misses, and hit_rate."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
        }

    @classmethod
    def from_env(cls, base_reranker):
        """Returns base_reranker unchanged if RERANKER_CACHE_REDIS_URL is not set.

        If the env var is set, attempts to connect to Redis and wrap the reranker.
        If connection fails, logs a warning and returns the base_reranker unwrapped.
        """
        redis_url = os.environ.get("RERANKER_CACHE_REDIS_URL")
        if not redis_url:
            return base_reranker
        try:
            import redis as _redis

            rc = _redis.from_url(redis_url)
            return cls(
                base_reranker,
                rc,
                ttl_seconds=int(os.environ.get("RERANKER_CACHE_TTL_SECONDS", "300")),
            )
        except Exception as exc:
            logger.warning("Reranker cache disabled: %s", exc)
            return base_reranker
