"""Redis-backed cache for query embeddings + OpenAI embedding provider.

Cache key: ``emb:{sha256(text)}``.
Value: raw numpy bytes prefixed with a 4-byte shape header (no pickle needed).

``EmbeddingCache`` — low-level Redis get/set.
``OpenAIEmbedder``  — high-level: Redis first, OpenAI on miss, write back.

Both classes degrade silently when Redis / OpenAI are unavailable.
"""

from __future__ import annotations

import hashlib
import logging
import struct

import numpy as np

logger = logging.getLogger(__name__)

# Header format: dtype char (1 byte) + ndim (1 byte) + shape ints (4 bytes each)
_DTYPE_CODES: dict[str, int] = {"float32": 0, "float64": 1, "float16": 2}
_CODE_DTYPES: dict[int, str] = {v: k for k, v in _DTYPE_CODES.items()}


def _pack(arr: np.ndarray) -> bytes:
    dtype_code = _DTYPE_CODES.get(arr.dtype.name, 0)
    header = struct.pack(
        f">BB{len(arr.shape)}I", dtype_code, len(arr.shape), *arr.shape
    )
    return header + arr.astype(np.float32).tobytes()


def _unpack(data: bytes) -> np.ndarray:
    dtype_code, ndim = struct.unpack_from(">BB", data, 0)
    shape = struct.unpack_from(f">{ndim}I", data, 2)
    payload_offset = 2 + ndim * 4
    dtype = _CODE_DTYPES.get(dtype_code, "float32")
    return (
        np.frombuffer(data[payload_offset:], dtype=np.float32)
        .reshape(shape)
        .astype(dtype)
    )


class EmbeddingCache:
    """Thin Redis wrapper for caching dense query embeddings.

    Args:
        redis_url: Redis connection URL, e.g. ``redis://localhost:6379/0``.
        ttl_seconds: How long to keep cached embeddings (default 7 days).
    """

    def __init__(
        self,
        *,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 7 * 24 * 3600,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self._client = None
        self._unavailable = False
        try:
            import redis

            self._client = redis.Redis.from_url(redis_url, socket_connect_timeout=1)
            self._client.ping()
        except Exception as exc:
            logger.warning(
                "Redis embedding cache unavailable: %s — skipping cache.", exc
            )
            self._unavailable = True

    def _key(self, text: str) -> str:
        return f"emb:{hashlib.sha256(text.encode()).hexdigest()}"

    def get(self, text: str) -> np.ndarray | None:
        if self._unavailable or self._client is None:
            return None
        try:
            data = self._client.get(self._key(text))
            return _unpack(data) if data else None
        except Exception as exc:
            logger.debug("Cache get failed: %s", exc)
            return None

    def set(self, text: str, embedding: np.ndarray) -> None:
        if self._unavailable or self._client is None:
            return
        try:
            self._client.setex(self._key(text), self.ttl_seconds, _pack(embedding))
        except Exception as exc:
            logger.debug("Cache set failed: %s", exc)

    def get_batch(self, texts: list[str]) -> list[np.ndarray | None]:
        """Return one entry per text; None = cache miss."""
        if self._unavailable or self._client is None:
            return [None] * len(texts)
        try:
            keys = [self._key(t) for t in texts]
            raw_values = self._client.mget(keys)
            return [_unpack(v) if v else None for v in raw_values]
        except Exception as exc:
            logger.debug("Cache mget failed: %s", exc)
            return [None] * len(texts)

    def set_batch(self, texts: list[str], embeddings: np.ndarray) -> None:
        """Write one embedding row per text."""
        if self._unavailable or self._client is None:
            return
        try:
            pipe = self._client.pipeline(transaction=False)
            for text, row in zip(texts, embeddings):
                pipe.setex(self._key(text), self.ttl_seconds, _pack(row))
            pipe.execute()
        except Exception as exc:
            logger.debug("Cache pipeline set failed: %s", exc)


class OpenAIEmbedder:
    """Embed texts via OpenAI API with Redis as a look-aside cache.

    Flow for every ``embed()`` call:
      1. ``mget`` all keys from Redis — return hits immediately.
      2. Call OpenAI only for cache misses (batched, ≤ 100 per request).
      3. ``pipeline setex`` new embeddings back to Redis.
      4. Return a stacked numpy array in original input order.

    Args:
        model: OpenAI embedding model name.
        redis_url: Redis URL.  ``None`` disables caching (always calls OpenAI).
        ttl_seconds: Redis TTL for stored embeddings.
        api_key: Override ``OPENAI_API_KEY`` env var.
    """

    _BATCH_SIZE = 100  # OpenAI recommends ≤ 2048 inputs; 100 keeps latency low

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        redis_url: str | None = None,
        ttl_seconds: int = 7 * 24 * 3600,
        api_key: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self._openai_client = OpenAI(
            api_key=api_key
        )  # created once; api_key=None → reads OPENAI_API_KEY from env
        self._cache = (
            EmbeddingCache(redis_url=redis_url, ttl_seconds=ttl_seconds)
            if redis_url
            else None
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (N, D) float32 array — one row per input text."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        cached: list[np.ndarray | None] = (
            self._cache.get_batch(texts) if self._cache else [None] * len(texts)
        )

        miss_indices = [i for i, v in enumerate(cached) if v is None]
        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            fresh = self._call_openai(miss_texts)
            if self._cache:
                self._cache.set_batch(miss_texts, fresh)
            for i, row in zip(miss_indices, fresh):
                cached[i] = row

        return np.stack(cached)  # type: ignore[arg-type]

    def _call_openai(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[start : start + self._BATCH_SIZE]
            response = self._openai_client.embeddings.create(
                input=batch, model=self.model
            )
            rows.extend(
                item.embedding for item in sorted(response.data, key=lambda x: x.index)
            )
        return np.array(rows, dtype=np.float32)
