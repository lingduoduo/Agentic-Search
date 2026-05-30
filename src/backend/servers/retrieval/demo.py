"""Lightweight demo retrieval server — no index, no Java, no pyserini.

Loads corpus.jsonl at startup and scores documents with TF-IDF.
Exposes the same /retrieve API as the full retrieval server so the
web frontend works out of the box for local demos.

Usage:
    python3 -m src.servers.retrieval.demo --corpus_path data/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json

from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .app import add_host_port_args, create_base_app, load_environment, run_uvicorn_app

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_TOPK = 5


def _load_corpus(corpus_path: str) -> list[dict]:
    docs = []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


class TfidfRetriever:
    def __init__(self, corpus_path: str) -> None:
        self._docs = _load_corpus(corpus_path)
        texts = [
            f"{d.get('title', '')} {d.get('contents', d.get('text', ''))}"
            for d in self._docs
        ]
        self._vec = TfidfVectorizer(stop_words="english")
        self._matrix = self._vec.fit_transform(texts)

    def retrieve(self, queries: list[str], topk: int) -> list[list[dict]]:
        results = []
        q_matrix = self._vec.transform(queries)
        scores = cosine_similarity(q_matrix, self._matrix)
        for row_scores in scores:
            ranked = sorted(enumerate(row_scores), key=lambda x: x[1], reverse=True)[
                :topk
            ]
            results.append(
                [
                    {
                        "document": {
                            "id": self._docs[i].get("id", str(i)),
                            "title": self._docs[i].get("title", ""),
                            "text": self._docs[i].get(
                                "contents", self._docs[i].get("text", "")
                            ),
                            "url": self._docs[i].get("url"),
                        },
                        "score": float(score),
                    }
                    for i, score in ranked
                ]
            )
        return results


class RetrieveRequest(BaseModel):
    queries: list[str] | None = None
    query: str | None = None
    topk: int = DEFAULT_TOPK
    return_scores: bool = False

    def resolved_queries(self) -> list[str]:
        if self.queries:
            return self.queries
        if self.query:
            return [self.query]
        return []


def create_app(retriever: TfidfRetriever):
    app = create_base_app("Demo Retrieval Server")

    @app.post("/retrieve")
    def retrieve_endpoint(body: RetrieveRequest):
        queries = body.resolved_queries()
        rows = retriever.retrieve(queries, topk=body.topk)
        if not body.return_scores:
            rows = [[item["document"] for item in row] for row in rows]
        if body.query is not None:
            return {"results": rows[0] if rows else []}
        return {"results": rows}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo TF-IDF retrieval server")
    parser.add_argument(
        "--corpus_path", type=str, required=True, help="Path to corpus.jsonl"
    )
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    add_host_port_args(
        parser,
        "DEMO_RETRIEVAL_HOST",
        "DEMO_RETRIEVAL_PORT",
        default_host=DEFAULT_HOST,
        default_port=DEFAULT_PORT,
    )
    return parser.parse_args()


def main() -> None:
    load_environment()
    args = parse_args()
    retriever = TfidfRetriever(args.corpus_path)
    app = create_app(retriever)
    run_uvicorn_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
