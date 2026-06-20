"""Assemble the query-transform wrapper chain from environment variables.

Chain (outermost → innermost):
    RoutedQueryTransformPipeline → CachedQueryTransformPipeline
        → AsyncQueryTransformPipeline → QueryTransformPipeline
Each layer is optional; unset env vars leave the chain unchanged.
RoutedQueryTransformPipeline is wired in Task 9.
"""

from __future__ import annotations

import os


def _flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def build_query_transform_pipeline_from_env(llm: object) -> object | None:
    from src.context.query_transform import QueryTransformPipeline

    leaf = QueryTransformPipeline.from_env(llm)
    if leaf is None:
        if not _flag("QT_ROUTER"):
            return None
        # Router can run standalone: build a default all-off leaf for it to route into.
        from src.context.query_transform import QueryTransformConfig

        leaf = QueryTransformPipeline(QueryTransformConfig(), llm)
    pipe: object = leaf
    if _flag("QT_ASYNC"):
        from src.internal.retrieval.async_query_transform import (
            AsyncQueryTransformPipeline,
        )

        pipe = AsyncQueryTransformPipeline.from_env(pipe)

    from src.internal.retrieval.cached_query_transform import (
        CachedQueryTransformPipeline,
    )

    pipe = CachedQueryTransformPipeline.from_env(
        pipe
    )  # returns pipe unchanged if no URL

    if _flag("QT_ROUTER"):
        from src.internal.retrieval.routed_query_transform import (
            RoutedQueryTransformPipeline,
        )

        pipe = RoutedQueryTransformPipeline.from_env(pipe)
    return pipe
