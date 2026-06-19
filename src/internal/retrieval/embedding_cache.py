"""Redis-backed embedding cache and async embedding batcher.

CachedEmbedder: single-query cache keyed by sha256(query). TTL: 1 hour.
EmbeddingBatcher: coalesces concurrent async embed() calls into batches.
"""

from __future__ import annotations

import asyncio
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


class EmbeddingBatcher:
    """Coalesces concurrent async embed() calls into a single encode_batch() call.

    Requests arriving within wait_ms of each other are grouped into one batch
    (up to max_batch). Reduces model invocations when many queries arrive
    concurrently (e.g. agent multi-step loops).

    base_embedder must implement: encode_batch(texts: list[str]) -> np.ndarray
    """

    def __init__(
        self,
        base_embedder: Any,
        *,
        max_batch: int = 32,
        wait_ms: float = 5.0,
    ) -> None:
        self._embedder = base_embedder
        self._max_batch = max_batch
        self._wait_s = wait_ms / 1000.0
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _setup_for_loop(self) -> None:
        """Reset state when a new event loop is detected (e.g. per-test loops)."""
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            self._loop = loop
            self._queue = asyncio.Queue()
            self._task = None

    def _ensure_running(self) -> None:
        self._setup_for_loop()
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._worker())

    async def embed(self, text: str) -> Any:
        self._ensure_running()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        assert self._queue is not None
        await self._queue.put((text, fut))
        return await fut

    async def _worker(self) -> None:
        assert self._queue is not None
        while True:
            text, fut = await self._queue.get()
            batch: list[tuple[str, asyncio.Future]] = [(text, fut)]

            try:
                loop = asyncio.get_running_loop()
                deadline = loop.time() + self._wait_s
                while len(batch) < self._max_batch:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    t, f = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append((t, f))
            except asyncio.TimeoutError:
                pass

            texts = [t for t, _ in batch]
            futures = [f for _, f in batch]
            try:
                vectors = self._embedder.encode_batch(texts)
                for i, f in enumerate(futures):
                    if not f.done():
                        f.set_result(vectors[i])
            except Exception as exc:
                for f in futures:
                    if not f.done():
                        f.set_exception(exc)
