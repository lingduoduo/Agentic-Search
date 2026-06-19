from __future__ import annotations

import dataclasses
import logging
import os

from src.internal.retrieval.backends.base import RetrievalResult

logger = logging.getLogger(__name__)


class ONNXReranker:
    """Drop-in Reranker replacement using ONNX runtime (requires optimum)."""

    def __init__(self, model_name: str, *, device: str = "cpu") -> None:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = ORTModelForSequenceClassification.from_pretrained(
            model_name, export=True
        )
        self._device = device

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        import torch

        pairs = [[query, f"{r.title}\n{r.text}"] for r in results]
        inputs = self._tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits.squeeze(-1)
        scores = logits.tolist() if hasattr(logits, "tolist") else list(logits)
        if isinstance(scores, float):
            scores = [scores]
        scored = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        return [dataclasses.replace(r, score=float(s)) for s, r in scored[:top_k]]

    @staticmethod
    def from_env():
        """Returns ONNXReranker or falls back to Reranker. Returns None if no provider set."""
        from src.internal.retrieval.reranker import Reranker

        if os.environ.get("RERANKER_USE_ONNX", "").lower() not in ("1", "true", "yes"):
            return Reranker.from_env()
        try:
            model = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
            return ONNXReranker(model, device=os.environ.get("RERANKER_DEVICE", "cpu"))
        except ImportError:
            logger.warning("optimum not installed; falling back to PyTorch Reranker")
            return Reranker.from_env()
