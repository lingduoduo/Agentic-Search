"""Utilities for reranking retrieved documents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np


def _require_torch():
    import torch

    return torch


def _require_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder


def passage_to_string(doc_item: dict[str, Any]) -> str:
    if "document" in doc_item and isinstance(doc_item["document"], dict):
        document = doc_item["document"]
        if "contents" in document:
            content = document["contents"]
        else:
            content = json.dumps(document, ensure_ascii=False)
    else:
        content = doc_item.get("contents", "")

    title, _, body = content.partition("\n")
    clean_title = title.strip().strip('"')
    if clean_title:
        return f"(Title: {clean_title}) {body}".strip()
    return content


def string_to_document(text: str) -> dict[str, dict[str, str]]:
    match = re.match(r"\(Title:\s*([^)]+)\)\s*(.+)", text, re.DOTALL)
    if match:
        title, content = match.groups()
        return {"document": {"contents": f'"{title}"\n{content}'}}
    return {"document": {"contents": text}}


@dataclass(frozen=True)
class RerankerConfig:
    model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L12-v2"
    batch_size: int = 32
    rerank_topk: int = 3
    reranker_type: str = "sentence_transformer"

    def validate(self) -> None:
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path is required.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.rerank_topk < 1:
            raise ValueError("rerank_topk must be at least 1.")
        if self.reranker_type != "sentence_transformer":
            raise ValueError(f"Unknown reranker type: {self.reranker_type}")


class SentenceTransformerReranker:
    def __init__(self, model: Any, batch_size: int = 32, device: str = "cpu"):
        self.model = model
        self.batch_size = batch_size
        self.device = device

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        torch = _require_torch()
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        if isinstance(scores, torch.Tensor) or isinstance(scores, np.ndarray):
            predicted = scores.tolist()
        else:
            predicted = list(scores)
        if len(predicted) != len(pairs):
            raise ValueError(
                "Reranker returned the wrong number of scores: "
                f"expected {len(pairs)}, got {len(predicted)}."
            )
        return predicted

    def rerank(
        self,
        queries: list[str],
        documents: list[list[dict[str, Any]]],
        topk: int | None = None,
    ) -> list[list[dict[str, Any]]]:
        if len(queries) != len(documents):
            raise ValueError("queries and documents must have the same length.")

        pairs: list[tuple[str, str]] = []
        index_map: list[tuple[int, int]] = []
        for query_index, (query, doc_list) in enumerate(zip(queries, documents)):
            for doc_index, doc_item in enumerate(doc_list):
                pairs.append((query, passage_to_string(doc_item)))
                index_map.append((query_index, doc_index))

        if not pairs:
            return [[] for _ in queries]

        scores = self._predict(pairs)
        grouped: list[list[dict[str, Any]]] = [[] for _ in queries]
        for score, (query_index, doc_index) in zip(scores, index_map):
            grouped[query_index].append(
                {
                    "document": documents[query_index][doc_index],
                    "score": float(score),
                }
            )

        for index, query_results in enumerate(grouped):
            query_results.sort(key=lambda item: item["score"], reverse=True)
            if topk is not None:
                grouped[index] = query_results[:topk]

        return grouped

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        *,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> "SentenceTransformerReranker":
        cross_encoder = _require_cross_encoder()
        model = cross_encoder(model_name_or_path, device=device)
        return cls(model=model, batch_size=batch_size, device=device)


def get_reranker(config: RerankerConfig) -> SentenceTransformerReranker:
    config.validate()
    torch = _require_torch()
    return SentenceTransformerReranker.load(
        config.model_name_or_path,
        batch_size=config.batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
