"""Reranker: single class supporting local cross-encoders and Cohere."""

from __future__ import annotations

import asyncio
import dataclasses
import os
from dataclasses import dataclass
from typing import Literal

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.servers.retrieval.rerank import SentenceTransformerReranker

try:
    from src.internal.natural_language_processing.search_nlp_models import (
        cohere_rerank_api,
    )
except ImportError:
    cohere_rerank_api = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RerankerConfig:
    provider: Literal["local", "cohere"]
    model: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 32
    device: str = "cpu"
    api_key: str | None = None
    top_k: int | None = None

    def validate(self) -> None:
        if self.provider not in ("local", "cohere"):
            raise ValueError(
                f"Unknown provider: {self.provider!r}. Use 'local' or 'cohere'."
            )
        if self.provider == "cohere" and not self.api_key:
            raise ValueError("api_key is required for provider='cohere'.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")


class Reranker:
    def __init__(self, config: RerankerConfig) -> None:
        config.validate()
        self._config = config
        if config.provider == "local":
            self._local = SentenceTransformerReranker.load(
                config.model,
                batch_size=config.batch_size,
                device=config.device,
            )

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Rescore results and return top_k sorted by descending score."""
        if not results:
            return results
        effective_k = self._config.top_k or top_k
        if self._config.provider == "local":
            return self._rerank_local(query, results, effective_k)
        return self._rerank_cohere(query, results, effective_k)

    def _rerank_local(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        docs = [
            {"contents": f"{r.title}\n{r.text}", "doc_id": r.doc_id} for r in results
        ]
        scored = self._local.rerank([query], [docs], topk=top_k)
        id_to_result = {r.doc_id: r for r in results}
        reranked = []
        for item in sorted(scored[0], key=lambda x: x["score"], reverse=True):
            doc_id = item["document"].get("doc_id")
            if doc_id and doc_id in id_to_result:
                reranked.append(
                    dataclasses.replace(
                        id_to_result[doc_id], score=float(item["score"])
                    )
                )
        return reranked[:top_k]

    def _rerank_cohere(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        passages = [r.text for r in results]
        scores = asyncio.run(
            cohere_rerank_api(query, passages, self._config.model, self._config.api_key)
        )
        scored = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        return [dataclasses.replace(r, score=float(s)) for s, r in scored[:top_k]]

    @classmethod
    def from_env(cls) -> "Reranker | None":
        """Build a Reranker from env vars. Returns None if RERANKER_PROVIDER is unset."""
        provider = os.environ.get("RERANKER_PROVIDER")
        if not provider:
            return None
        top_k_raw = os.environ.get("RERANKER_TOP_K")
        return cls(
            RerankerConfig(
                provider=provider,  # type: ignore[arg-type]
                model=os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
                batch_size=int(os.environ.get("RERANKER_BATCH_SIZE", "32")),
                device=os.environ.get("RERANKER_DEVICE", "cpu"),
                api_key=os.environ.get("COHERE_API_KEY"),
                top_k=int(top_k_raw) if top_k_raw else None,
            )
        )
