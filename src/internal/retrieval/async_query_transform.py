from __future__ import annotations

import concurrent.futures
import logging
import os

from src.context.query_transform import QueryTransformConfig

logger = logging.getLogger(__name__)


class AsyncQueryTransformPipeline:
    """Wraps a leaf QueryTransformPipeline; runs transform jobs in parallel."""

    def __init__(self, base, *, timeout_ms: int = 400, max_workers: int = 5) -> None:
        self._base = base
        self._timeout_ms = timeout_ms
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    @property
    def max_variants(self) -> int:
        return self._base.max_variants

    @property
    def base_config(self) -> QueryTransformConfig:
        return self._base.base_config

    def transform(self, query, filters=None, *, config_override=None):
        config = config_override or self._base.base_config
        jobs = self._base._build_jobs(query, config)
        futures = {field: self._executor.submit(fn) for field, fn in jobs.items()}
        results: dict = {}
        for field, fut in futures.items():
            try:
                results[field] = fut.result(timeout=self._timeout_ms / 1000)
            except Exception as exc:  # timeout or transform error → degrade field
                logger.warning("transform %s failed/timed out: %s", field, exc)
                fut.cancel()
        return self._base._assemble(query, results, filters)

    @classmethod
    def from_env(cls, base) -> "AsyncQueryTransformPipeline":
        return cls(
            base,
            timeout_ms=int(os.environ.get("QT_TRANSFORM_TIMEOUT_MS", "400")),
            max_workers=int(os.environ.get("QT_MAX_WORKERS", "5")),
        )
