"""Utilities for querying dense retrieval indexes."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

# Must be set before torch/faiss are imported to prevent an OpenMP conflict on macOS
# when both libraries bundle their own libomp.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from .index_builder import (
    _encode_batch,
    load_corpus,
    load_model,
    prepare_texts,
    resolve_pooling_method,
    _require_faiss,
    _require_torch,
)


@dataclass(frozen=True)
class DenseRetrieverConfig:
    model_path: str
    index_path: str
    corpus_path: str
    retrieval_method: str
    topk: int = 5
    max_length: int = 180
    use_fp16: bool = False
    pooling_method: str | None = None

    def validate(self) -> None:
        if not self.model_path:
            raise ValueError("model_path is required.")
        if not self.index_path:
            raise ValueError("index_path is required.")
        if not self.corpus_path:
            raise ValueError("corpus_path is required.")
        if not self.retrieval_method:
            raise ValueError("retrieval_method is required.")
        if self.topk < 1:
            raise ValueError("topk must be at least 1.")


class DenseRetriever:
    """Loads a FAISS index and returns the top matching corpus documents."""

    def __init__(self, config: DenseRetrieverConfig):
        config.validate()
        self.config = config
        torch = _require_torch()
        faiss = _require_faiss()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pooling_method = resolve_pooling_method(
            config.retrieval_method,
            config.pooling_method,
        )
        self.model, self.tokenizer = load_model(
            model_path=config.model_path,
            use_fp16=config.use_fp16,
            device=self.device,
        )
        self.index = faiss.read_index(config.index_path)
        self.corpus = load_corpus(config.corpus_path)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        torch = _require_torch()
        non_empty = [q.strip() for q in queries if q.strip()]
        if not non_empty:
            return np.empty((0, 0), dtype=np.float32)

        texts = prepare_texts(non_empty, self.config.retrieval_method, is_query=True)
        with torch.no_grad():
            return _encode_batch(
                self.model,
                self.tokenizer,
                texts,
                self.config.retrieval_method.lower(),
                self.config.max_length,
                self.pooling_method,
                self.device,
            )

    def retrieve(self, queries: list[str], topk: int | None = None) -> list[list[dict[str, Any]]]:
        clean_queries = [query.strip() for query in queries]
        query_embeddings = self.encode_queries(clean_queries)
        if query_embeddings.size == 0:
            return [[] for _ in clean_queries]

        scores, indices = self.index.search(query_embeddings, topk or self.config.topk)
        filled_results: list[list[dict[str, Any]]] = []
        for row_scores, row_indices in zip(scores, indices):
            query_results: list[dict[str, Any]] = []
            for score, idx in zip(row_scores, row_indices):
                if idx < 0:
                    continue
                item = self.corpus[int(idx)]
                query_results.append(
                    {
                        "document": item,
                        "score": float(score),
                    }
                )
            filled_results.append(query_results)

        filled_iter = iter(filled_results)
        return [[] if not query else next(filled_iter) for query in clean_queries]

    def batch_search(
        self,
        query_list: list[str],
        num: int | None = None,
        return_score: bool = False,
    ) -> list[list[dict[str, Any]]] | tuple[list[list[dict[str, Any]]], list[list[float]]]:
        results = self.retrieve(query_list, topk=num)
        if not return_score:
            return [[item["document"] for item in row] for row in results]
        scores = [[float(item["score"]) for item in row] for row in results]
        documents = [[item["document"] for item in row] for row in results]
        return documents, scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a dense retrieval index.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--retrieval_method", type=str, required=True)
    parser.add_argument("--queries", nargs="+", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=180)
    parser.add_argument("--use_fp16", default=False, action="store_true")
    parser.add_argument("--pooling_method", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = DenseRetriever(
        DenseRetrieverConfig(
            model_path=args.model_path,
            index_path=args.index_path,
            corpus_path=args.corpus_path,
            retrieval_method=args.retrieval_method,
            topk=args.topk,
            max_length=args.max_length,
            use_fp16=args.use_fp16,
            pooling_method=args.pooling_method,
        )
    )
    print(json.dumps(retriever.retrieve(args.queries), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
