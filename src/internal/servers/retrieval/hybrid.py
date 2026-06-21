"""Hybrid retrieval server — RRF-fused dense (e5, in-memory) + sparse (TF-IDF).

Java-free and FAISS-free. Exposes the same /retrieve API as demo.py so the web
backend's "Local Retrieval" provider gets hybrid results with no changes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np


logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
DEFAULT_MODEL = "intfloat/e5-base-v2"


def _passage_text(doc: dict) -> str:
    body = doc.get("contents", doc.get("text", ""))
    return f"passage: {doc.get('title', '')} {body}".strip()


def _as_document(doc: dict, index: int) -> dict:
    return {
        "id": doc.get("id", str(index)),
        "title": doc.get("title", ""),
        "text": doc.get("contents", doc.get("text", "")),
        "url": doc.get("url"),
    }


class DenseEmbeddingRetriever:
    """Dense retrieval over an in-memory e5 embedding matrix (no FAISS).

    Embeds corpus passages once at construction; retrieve() encodes each query
    as "query: <q>" and ranks documents by dot product (embeddings are
    L2-normalized, so dot product == cosine similarity).
    """

    def __init__(
        self, docs: list[dict], *, encoder: Callable[[list[str]], np.ndarray]
    ) -> None:
        self._docs = docs
        self._encoder = encoder
        if docs:
            self._matrix = encoder([_passage_text(d) for d in docs])
        else:
            self._matrix = np.empty((0, 0), dtype=np.float32)

    def retrieve(self, queries: list[str], topk: int) -> list[list[dict]]:
        if self._matrix.size == 0:
            return [[] for _ in queries]
        query_vecs = self._encoder([f"query: {q}" for q in queries])
        sims = query_vecs @ self._matrix.T
        results: list[list[dict]] = []
        for row in sims:
            ranked = sorted(enumerate(row), key=lambda x: x[1], reverse=True)[:topk]
            results.append(
                [
                    {"document": _as_document(self._docs[i], i), "score": float(score)}
                    for i, score in ranked
                ]
            )
        return results


def build_e5_encoder(
    model_name: str = DEFAULT_MODEL, device: str = "mps"
) -> Callable[[list[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

    return encode
