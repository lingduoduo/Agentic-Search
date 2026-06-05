"""Redis-backed cache for query embeddings.

Cache key: ``emb:{sha256(model_path + ":" + text)}``.
Value: raw numpy bytes prefixed with a 4-byte shape header so no extra
dependencies (pickle, msgpack) are needed.

Falls back silently when Redis is unavailable — retrieval continues
without caching and a warning is logged once on first failure.
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
