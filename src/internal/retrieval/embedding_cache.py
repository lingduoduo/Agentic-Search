"""Redis-backed embedding cache for query vectors.

Cache key: sha256(query)[:16]. TTL: 1 hour (3600s).
A cache hit skips the embedding call entirely (saves 30-80ms per query).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _cache_key(query: str) -> str:
    return f"emb:{hashlib.sha256(query.encode()).hexdigest()[:16]}"


class CachedEmbedder:
    """Wraps any embedder with a Redis cache keyed by sha256(query).

    redis_client: a redis.Redis instance, or None to disable caching.
    """

    def __init__(self, base_embedder: Any, redis_client: Any | None = None) -> None:
        self._embedder = base_embedder
        self._redis = redis_client

    def embed(self, query: str) -> list[float]:
        if self._redis is not None:
            key = _cache_key(query)
            cached = self._redis.get(key)
            if cached is not None:
                return json.loads(cached)

        vec: list[float] = self._embedder.encode(
            query, normalize_embeddings=True
        ).tolist()

        if self._redis is not None:
            self._redis.setex(key, 3600, json.dumps(vec))

        return vec
