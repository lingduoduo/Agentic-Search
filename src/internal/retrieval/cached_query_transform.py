"""Redis-backed cache for TransformedQueryBundle keyed by query + config signature."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from typing import Any

from src.context.query_transform import (
    QueryTransformConfig,
    TransformedQueryBundle,
    config_signature,
)

logger = logging.getLogger(__name__)


def _key(query: str, sig: str) -> str:
    raw = f"{query.lower().strip()}|{sig}"
    return "qt:" + hashlib.sha256(raw.encode()).hexdigest()[:20]


class CachedQueryTransformPipeline:
    """Redis cache of TransformedQueryBundle keyed by query + config signature."""

    def __init__(
        self, base, redis_client: Any | None = None, *, ttl_seconds: int = 600
    ):
        self._base = base
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    @property
    def max_variants(self) -> int:
        return self._base.max_variants

    @property
    def base_config(self) -> QueryTransformConfig:
        return self._base.base_config

    def transform(self, query, filters=None, *, config_override=None):
        config = config_override or self._base.base_config
        key = _key(query, config_signature(config))

        if self._redis is not None:
            raw = self._redis.get(key)
            if raw is not None:
                self._hits += 1
                data = json.loads(raw)
                bundle = TransformedQueryBundle(**data)
                # Re-merge caller filters (not part of the cached transform output).
                if filters:
                    return dataclasses.replace(
                        bundle, merged_filters={**bundle.merged_filters, **filters}
                    )
                return bundle

        self._misses += 1
        bundle = self._base.transform(query, filters, config_override=config_override)

        if self._redis is not None:
            self._redis.setex(key, self._ttl, json.dumps(dataclasses.asdict(bundle)))
        return bundle

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
        }

    @classmethod
    def from_env(cls, base):
        url = os.environ.get("QT_CACHE_REDIS_URL")
        if not url:
            return base
        try:
            import redis

            client = redis.Redis.from_url(url)
        except Exception as exc:  # pragma: no cover - infra dependent
            logger.warning("QT cache disabled, redis unavailable: %s", exc)
            return base
        return cls(
            base, client, ttl_seconds=int(os.environ.get("QT_CACHE_TTL_SECONDS", "600"))
        )
