from __future__ import annotations

from dataclasses import replace

from src.context.query_transform import QueryTransformConfig


class RoutedQueryTransformPipeline:
    """Outer wrapper: picks per-query transforms via a router, threads config down."""

    def __init__(self, base, router) -> None:
        self._base = base
        self._router = router

    @property
    def max_variants(self) -> int:
        return self._base.max_variants

    @property
    def base_config(self) -> QueryTransformConfig:
        return self._base.base_config

    def transform(self, query, filters=None, *, config_override=None):
        config = config_override or self._router.predict(query)
        # Preserve the configured max_variants from the base.
        config = replace(config, max_variants=self._base.base_config.max_variants)
        return self._base.transform(query, filters, config_override=config)

    @classmethod
    def from_env(cls, base):
        from src.internal.retrieval.query_router import QueryRouter

        router = QueryRouter.from_env()
        if router is None:
            return base
        return cls(base, router)
