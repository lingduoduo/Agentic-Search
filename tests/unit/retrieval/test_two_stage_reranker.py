from __future__ import annotations

import asyncio
from unittest.mock import MagicMock


from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.two_stage_reranker import TwoStageReranker


def _result(doc_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="t", text="c", url=None, score=score)


def _make_fast(return_val):
    m = MagicMock()
    m.rerank.return_value = return_val
    return m


def _make_heavy(return_val):
    m = MagicMock()
    m.rerank.return_value = return_val
    return m


def test_fast_gets_all_candidates():
    """Fast reranker receives all input results."""
    fast = _make_fast([_result("d1"), _result("d2")])
    heavy = _make_heavy([_result("d1")])
    tsr = TwoStageReranker(fast, heavy, pre_filter_top_n=2)
    inputs = [_result("d1"), _result("d2"), _result("d3")]
    tsr.rerank("q", inputs, top_k=1)
    fast.rerank.assert_called_once_with("q", inputs, 2)


def test_heavy_gets_fast_output():
    """Heavy reranker receives fast reranker's output."""
    fast_out = [_result("d2"), _result("d1")]
    fast = _make_fast(fast_out)
    heavy = _make_heavy([_result("d2")])
    tsr = TwoStageReranker(fast, heavy, pre_filter_top_n=2)
    tsr.rerank("q", [_result("d1"), _result("d2")], top_k=1)
    heavy.rerank.assert_called_once_with("q", fast_out, 1)


def test_returns_heavy_output():
    fast = _make_fast([_result("d1"), _result("d2")])
    heavy = _make_heavy([_result("d1")])
    tsr = TwoStageReranker(fast, heavy, pre_filter_top_n=2)
    result = tsr.rerank("q", [_result("d1"), _result("d2")], top_k=1)
    assert [r.doc_id for r in result] == ["d1"]


def test_empty_inputs_returns_empty():
    fast = _make_fast([])
    heavy = _make_heavy([])
    tsr = TwoStageReranker(fast, heavy, pre_filter_top_n=5)
    assert tsr.rerank("q", [], top_k=3) == []


def test_async_rerank_falls_back_to_sync_when_no_arerank():
    """When arerank is not available, fallback to sync rerank."""
    fast = _make_fast([_result("d1")])
    heavy = _make_heavy([_result("d1")])
    # Explicitly mark that arerank is not available on these mocks
    del fast.arerank
    del heavy.arerank
    tsr = TwoStageReranker(fast, heavy, pre_filter_top_n=2)
    result = asyncio.run(tsr.arerank("q", [_result("d1")], top_k=1))
    assert result[0].doc_id == "d1"
    # Verify sync rerank was called, not arerank
    fast.rerank.assert_called_once_with("q", [_result("d1")], 2)
    heavy.rerank.assert_called_once_with("q", [_result("d1")], 1)


def test_from_env_reads_pre_filter_top_n(monkeypatch):
    monkeypatch.setenv("RERANKER_PRE_FILTER_TOP_N", "25")
    fast = MagicMock()
    heavy = MagicMock()
    tsr = TwoStageReranker.from_env(fast, heavy)
    assert tsr._pre_n == 25
