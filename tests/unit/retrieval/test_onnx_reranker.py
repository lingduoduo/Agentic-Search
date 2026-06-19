from __future__ import annotations

import pytest

optimum = pytest.importorskip("optimum")  # noqa: E402

from unittest.mock import MagicMock, patch  # noqa: E402

from src.internal.retrieval.backends.base import RetrievalResult  # noqa: E402
from src.internal.retrieval.onnx_reranker import ONNXReranker  # noqa: E402


def _result(doc_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="t", text="c", url=None, score=score)


def _make_onnx_reranker():
    with (
        patch(
            "optimum.onnxruntime.ORTModelForSequenceClassification.from_pretrained",
            return_value=MagicMock(),
        ),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
    ):
        return ONNXReranker("BAAI/bge-reranker-base")


def test_onnx_reranker_rerank_returns_results():
    reranker = _make_onnx_reranker()
    # Patch the model to return fake logits
    import torch

    reranker._model.return_value.logits = torch.tensor([[0.8], [0.3]])
    results = [_result("d1"), _result("d2")]
    out = reranker.rerank("query", results, top_k=2)
    assert len(out) == 2
    assert out[0].doc_id == "d1"  # d1 has higher score


def test_onnx_reranker_respects_top_k():
    reranker = _make_onnx_reranker()
    import torch

    reranker._model.return_value.logits = torch.tensor([[0.9], [0.5], [0.1]])
    results = [_result("d1"), _result("d2"), _result("d3")]
    out = reranker.rerank("query", results, top_k=2)
    assert len(out) == 2


def test_from_env_returns_reranker_when_onnx_disabled(monkeypatch):
    monkeypatch.delenv("RERANKER_USE_ONNX", raising=False)
    from src.internal.retrieval.onnx_reranker import ONNXReranker
    from src.internal.retrieval.reranker import Reranker

    monkeypatch.delenv("RERANKER_PROVIDER", raising=False)
    result = ONNXReranker.from_env()
    assert result is None or isinstance(result, Reranker)
