from __future__ import annotations

import asyncio
import concurrent.futures
import os

from src.internal.retrieval.backends.base import RetrievalResult


class RerankerTimeoutError(RuntimeError):
    pass


class AsyncReranker:
    """Wraps any reranker, offloads scoring to a thread pool with a timeout."""

    def __init__(
        self,
        base_reranker,
        *,
        timeout_ms: int = 500,
        max_workers: int = 4,
    ) -> None:
        self._base = base_reranker
        self._timeout_ms = timeout_ms
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Sync shim: submits to thread pool, blocks with timeout."""
        future = self._executor.submit(self._base.rerank, query, results, top_k)
        try:
            return future.result(timeout=self._timeout_ms / 1000)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise RerankerTimeoutError(
                f"Reranker exceeded {self._timeout_ms}ms timeout"
            )

    async def arerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Async entry point: runs scorer in thread pool, awaits with timeout."""
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._executor, self._base.rerank, query, results, top_k
        )
        try:
            return await asyncio.wait_for(future, timeout=self._timeout_ms / 1000)
        except asyncio.TimeoutError:
            raise RerankerTimeoutError(
                f"Reranker exceeded {self._timeout_ms}ms timeout"
            )

    @classmethod
    def from_env(cls, base_reranker) -> AsyncReranker:
        return cls(
            base_reranker,
            timeout_ms=int(os.environ.get("RERANKER_TIMEOUT_MS", "500")),
            max_workers=int(os.environ.get("RERANKER_MAX_WORKERS", "4")),
        )
