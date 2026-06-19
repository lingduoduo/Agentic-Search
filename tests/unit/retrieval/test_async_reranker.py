from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from src.internal.retrieval.async_reranker import AsyncReranker, RerankerTimeoutError
from src.internal.retrieval.backends.base import RetrievalResult


def _result(doc_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="t", text="c", url=None, score=score)


def _make_base(return_val=None):
    base = MagicMock()
    base.rerank.return_value = return_val or [_result("d1")]
    return base


def test_sync_rerank_returns_results():
    base = _make_base([_result("d1"), _result("d2")])
    ar = AsyncReranker(base, timeout_ms=1000)
    results = ar.rerank("query", [_result("d1"), _result("d2")], top_k=2)
    assert [r.doc_id for r in results] == ["d1", "d2"]


def test_sync_rerank_delegates_to_base():
    base = _make_base()
    ar = AsyncReranker(base, timeout_ms=1000)
    ar.rerank("q", [_result("x")], top_k=1)
    base.rerank.assert_called_once_with("q", [_result("x")], 1)


def test_sync_rerank_timeout_raises():
    import time as _time

    base = MagicMock()

    def slow(*_):
        _time.sleep(0.3)
        return [_result("d1")]

    base.rerank.side_effect = slow
    ar = AsyncReranker(base, timeout_ms=50)
    with pytest.raises(RerankerTimeoutError):
        ar.rerank("q", [_result("d1")], top_k=1)


def test_async_rerank_returns_results():
    base = _make_base([_result("d1")])
    ar = AsyncReranker(base, timeout_ms=1000)
    results = asyncio.run(ar.arerank("q", [_result("d1")], top_k=1))
    assert results[0].doc_id == "d1"


def test_async_rerank_timeout_raises():
    import time as _time

    base = MagicMock()

    def slow(*_):
        _time.sleep(0.3)
        return [_result("d1")]

    base.rerank.side_effect = slow
    ar = AsyncReranker(base, timeout_ms=50)
    with pytest.raises(RerankerTimeoutError):
        asyncio.run(ar.arerank("q", [_result("d1")], top_k=1))


def test_from_env_reads_timeout(monkeypatch):
    monkeypatch.setenv("RERANKER_TIMEOUT_MS", "250")
    base = _make_base()
    ar = AsyncReranker.from_env(base)
    assert ar._timeout_ms == 250
