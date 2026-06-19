"""Tests for build_reranker_from_env factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from src.internal.retrieval.reranker_factory import build_reranker_from_env


def _mock_reranker():
    m = MagicMock()
    m.rerank.return_value = []
    return m


@patch.dict("os.environ", {}, clear=False)
def test_returns_none_when_provider_unset(monkeypatch):
    monkeypatch.delenv("RERANKER_PROVIDER", raising=False)
    assert build_reranker_from_env() is None


_RERANKER_CLS = "src.internal.retrieval.reranker.Reranker"


@patch(_RERANKER_CLS)
def test_returns_base_reranker_by_default(mock_cls, monkeypatch):
    monkeypatch.setenv("RERANKER_PROVIDER", "local")
    monkeypatch.delenv("RERANKER_ASYNC", raising=False)
    monkeypatch.delenv("RERANKER_TWO_STAGE", raising=False)
    base = _mock_reranker()
    mock_cls.from_env.return_value = base
    result = build_reranker_from_env()
    assert result is base


@patch(_RERANKER_CLS)
def test_wraps_in_async_cache_when_flag_set(mock_cls, monkeypatch):
    monkeypatch.setenv("RERANKER_PROVIDER", "local")
    monkeypatch.setenv("RERANKER_ASYNC", "true")
    monkeypatch.delenv("RERANKER_TWO_STAGE", raising=False)
    base = _mock_reranker()
    mock_cls.from_env.return_value = base

    from src.internal.retrieval.async_reranker import AsyncReranker
    from src.internal.retrieval.cached_reranker import CachedReranker

    with (
        patch.object(
            AsyncReranker, "from_env", return_value=_mock_reranker()
        ) as async_fe,
        patch.object(
            CachedReranker, "from_env", return_value=_mock_reranker()
        ) as cache_fe,
    ):
        result = build_reranker_from_env()
        async_fe.assert_called_once_with(base)
        cache_fe.assert_called_once()
        assert result is cache_fe.return_value


@patch(_RERANKER_CLS)
def test_two_stage_uses_same_model_when_fast_model_unset(mock_cls, monkeypatch):
    monkeypatch.setenv("RERANKER_PROVIDER", "local")
    monkeypatch.setenv("RERANKER_TWO_STAGE", "true")
    monkeypatch.delenv("RERANKER_ASYNC", raising=False)
    monkeypatch.delenv("RERANKER_FAST_MODEL", raising=False)
    base = _mock_reranker()
    mock_cls.from_env.return_value = base

    from src.internal.retrieval.two_stage_reranker import TwoStageReranker

    with patch.object(
        TwoStageReranker, "from_env", return_value=_mock_reranker()
    ) as ts_fe:
        result = build_reranker_from_env()
        ts_fe.assert_called_once_with(base, base)
        assert result is ts_fe.return_value


@patch(_RERANKER_CLS)
def test_two_stage_builds_separate_fast_reranker(mock_cls, monkeypatch):
    monkeypatch.setenv("RERANKER_PROVIDER", "local")
    monkeypatch.setenv("RERANKER_TWO_STAGE", "true")
    monkeypatch.setenv("RERANKER_FAST_MODEL", "BAAI/bge-reranker-base")
    monkeypatch.delenv("RERANKER_ASYNC", raising=False)
    heavy = _mock_reranker()
    fast = _mock_reranker()
    mock_cls.from_env.return_value = heavy
    mock_cls.return_value = fast

    from src.internal.retrieval.two_stage_reranker import TwoStageReranker

    with patch.object(
        TwoStageReranker, "from_env", return_value=_mock_reranker()
    ) as ts_fe:
        build_reranker_from_env()
        ts_fe.assert_called_once_with(fast, heavy)
