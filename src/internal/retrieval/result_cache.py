"""Redis-backed cache for full retrieval result lists.

Cache key: sha256(canonical_query + json(sorted_filters) + str(top_k))[:20].
A cache hit skips both BM25 and dense retrieval legs entirely.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from typing import Any

from .backends.base import RetrievalResult

logger = logging.getLogger(__name__)


def _cache_key(query: str, filters: dict | None, top_k: int) -> str:
    canonical = query.lower().strip()
    filters_str = json.dumps(filters, sort_keys=True) if filters else ""
    raw = f"res:{canonical}:{filters_str}:{top_k}"
    return "rrc:" + hashlib.sha256(raw.encode()).hexdigest()[:20]


class ResultCache:
    """Caches full retrieval result lists in Redis.

    Args:
        redis_client: A redis.Redis instance, or None to disable.
        ttl_seconds:  Cache entry TTL (default 300s).
    """

    def __init__(self, redis_client: Any | None, *, ttl_seconds: int = 300) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def get(
        self, query: str, filters: dict | None, top_k: int
    ) -> list[RetrievalResult] | None:
        if self._redis is None:
            self._misses += 1
            return None
        key = _cache_key(query, filters, top_k)
        raw = self._redis.get(key)
        if raw is None:
            self._misses += 1
            return None
        self._hits += 1
        rows = json.loads(raw)
        return [RetrievalResult(**row) for row in rows]

    def set(
        self,
        query: str,
        filters: dict | None,
        top_k: int,
        results: list[RetrievalResult],
    ) -> None:
        if self._redis is None:
            return
        key = _cache_key(query, filters, top_k)
        payload = json.dumps([asdict(r) for r in results])
        self._redis.setex(key, self._ttl, payload)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
        }
