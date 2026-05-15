"""Utilities for querying sparse BM25 retrieval indexes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from .index_builder import load_corpus


def _require_lucene_searcher():
    from pyserini.search.lucene import LuceneSearcher

    return LuceneSearcher


@dataclass(frozen=True)
class SparseRetrieverConfig:
    index_path: str
    corpus_path: str
    retrieval_method: str = "bm25"
    topk: int = 5

    def validate(self) -> None:
        if not self.index_path:
            raise ValueError("index_path is required.")
        if not self.corpus_path:
            raise ValueError("corpus_path is required.")
        if self.retrieval_method.lower() != "bm25":
            raise ValueError(
                "SparseRetriever currently supports retrieval_method='bm25'."
            )
        if self.topk < 1:
            raise ValueError("topk must be at least 1.")


class SparseRetriever:
    """Loads a local Pyserini BM25 index and returns matching corpus documents."""

    def __init__(self, config: SparseRetrieverConfig):
        config.validate()
        self.config = config
        lucene_searcher = _require_lucene_searcher()
        self.searcher = lucene_searcher(config.index_path)
        self._corpus_by_id: dict[str, dict[str, Any]] | None = None

    def _ensure_corpus_by_id(self) -> dict[str, dict[str, Any]]:
        if self._corpus_by_id is not None:
            return self._corpus_by_id

        corpus = load_corpus(self.config.corpus_path)
        corpus_by_id: dict[str, dict[str, Any]] = {}
        for index in range(len(corpus)):
            item = corpus[index]
            item_id = item.get("id")
            if item_id is not None:
                corpus_by_id[str(item_id)] = item
        self._corpus_by_id = corpus_by_id
        return corpus_by_id

    def _document_for_hit(self, hit: Any) -> dict[str, Any]:
        raw_doc = self.searcher.doc(hit.docid)
        raw = raw_doc.raw() if raw_doc is not None else ""
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {"id": hit.docid, "contents": raw}
            if isinstance(parsed, dict):
                return parsed

        corpus_item = self._ensure_corpus_by_id().get(str(hit.docid))
        if corpus_item is not None:
            return corpus_item
        return {"id": hit.docid, "contents": ""}

    def retrieve(
        self, queries: list[str], topk: int | None = None
    ) -> list[list[dict[str, Any]]]:
        resolved_topk = topk if topk is not None else self.config.topk
        rows: list[list[dict[str, Any]]] = []
        for query in queries:
            clean_query = query.strip()
            if not clean_query:
                rows.append([])
                continue
            hits = self.searcher.search(clean_query, k=resolved_topk)
            rows.append(
                [
                    {
                        "document": self._document_for_hit(hit),
                        "score": float(hit.score),
                    }
                    for hit in hits
                ]
            )
        return rows

    def batch_search(
        self,
        query_list: list[str],
        num: int | None = None,
        return_score: bool = False,
    ) -> (
        list[list[dict[str, Any]]]
        | tuple[list[list[dict[str, Any]]], list[list[float]]]
    ):
        results = self.retrieve(query_list, topk=num)
        if not return_score:
            return [[item["document"] for item in row] for row in results]
        documents, scores = [], []
        for row in results:
            documents.append([item["document"] for item in row])
            scores.append([float(item["score"]) for item in row])
        return documents, scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a sparse BM25 retrieval index.")
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--queries", nargs="+", required=True)
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = SparseRetriever(
        SparseRetrieverConfig(
            index_path=args.index_path,
            corpus_path=args.corpus_path,
            topk=args.topk,
        )
    )
    print(json.dumps(retriever.retrieve(args.queries), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
