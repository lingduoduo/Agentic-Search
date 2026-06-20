from __future__ import annotations

from unittest.mock import MagicMock

from src.internal.retrieval.query_transform_factory import (
    build_query_transform_pipeline_from_env,
)


def test_returns_none_when_all_flags_unset(monkeypatch):
    for v in (
        "QT_DECOMPOSE",
        "QT_HYDE",
        "QT_STEP_BACK",
        "QT_KEYWORDS",
        "QT_CONSTRUCT_FILTERS",
        "QT_ASYNC",
        "QT_ROUTER",
    ):
        monkeypatch.delenv(v, raising=False)
    assert build_query_transform_pipeline_from_env(MagicMock()) is None


def test_multi_query_alone_builds_pipeline(monkeypatch):
    for v in (
        "QT_DECOMPOSE",
        "QT_HYDE",
        "QT_STEP_BACK",
        "QT_KEYWORDS",
        "QT_CONSTRUCT_FILTERS",
        "QT_ASYNC",
        "QT_ROUTER",
        "QT_CACHE_REDIS_URL",
    ):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("QT_MULTI_QUERY", "true")
    assert build_query_transform_pipeline_from_env(MagicMock()) is not None


def test_router_alone_builds_routed_pipeline(monkeypatch):
    for v in (
        "QT_DECOMPOSE",
        "QT_HYDE",
        "QT_STEP_BACK",
        "QT_KEYWORDS",
        "QT_CONSTRUCT_FILTERS",
        "QT_MULTI_QUERY",
        "QT_ASYNC",
        "QT_CACHE_REDIS_URL",
    ):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("QT_ROUTER", "true")
    pipe = build_query_transform_pipeline_from_env(MagicMock())
    assert type(pipe).__name__ == "RoutedQueryTransformPipeline"


def test_async_wraps_leaf(monkeypatch):
    monkeypatch.setenv("QT_STEP_BACK", "true")
    monkeypatch.setenv("QT_ASYNC", "true")
    monkeypatch.delenv("QT_CACHE_REDIS_URL", raising=False)
    monkeypatch.delenv("QT_ROUTER", raising=False)
    pipe = build_query_transform_pipeline_from_env(MagicMock())
    assert type(pipe).__name__ == "AsyncQueryTransformPipeline"
