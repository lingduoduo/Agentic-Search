"""FastAPI server that combines dense retrieval with cross-encoder reranking."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass

import uvicorn
from fastapi import HTTPException
from pydantic import BaseModel, Field

from ..rerank import RerankerConfig, get_reranker
from ..dense_retriever import DenseRetriever, DenseRetrieverConfig
from .app import create_base_app

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


@dataclass(frozen=True)
class RetrievalRerankConfig:
    retriever: DenseRetrieverConfig
    reranker: RerankerConfig


class SearchRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1)
    topk_retrieval: int | None = None
    topk_rerank: int | None = None
    return_scores: bool = False


def create_app(config: RetrievalRerankConfig):
    retriever = DenseRetriever(config.retriever)
    reranker = get_reranker(config.reranker)
    app = create_base_app("Retrieval and Rerank Server")

    @app.post("/retrieve")
    def search_endpoint(request: SearchRequest) -> dict[str, list]:
        try:
            retrieved = retriever.retrieve(
                request.queries,
                topk=request.topk_retrieval or config.retriever.topk,
            )
            reranked = reranker.rerank(
                request.queries,
                [[item["document"] for item in row] for row in retrieved],
                topk=request.topk_rerank or config.reranker.rerank_topk,
            )
        except Exception as exc:
            logger.exception("Retrieval+rerank request failed: %s", exc)
            raise HTTPException(
                status_code=500, detail="Retrieval+rerank request failed"
            ) from exc

        if request.return_scores:
            return {"result": reranked}
        return {
            "result": [
                [item["document"] for item in query_results]
                for query_results in reranked
            ]
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the local retrieval+rerank server."
    )
    parser.add_argument(
        "--index_path", type=str, required=True, help="Corpus index file."
    )
    parser.add_argument(
        "--corpus_path", type=str, required=True, help="Local corpus file."
    )
    parser.add_argument("--retrieval_topk", type=int, default=10)
    parser.add_argument("--retrieval_method", type=str, required=True)
    parser.add_argument("--retriever_model", type=str, required=True)
    parser.add_argument("--retrieval_max_length", type=int, default=180)
    parser.add_argument("--retrieval_use_fp16", action="store_true", default=False)
    parser.add_argument("--retrieval_pooling_method", type=str, default=None)
    parser.add_argument("--rerank_topk", type=int, default=3)
    parser.add_argument(
        "--reranker_model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L12-v2",
    )
    parser.add_argument("--reranker_batch_size", type=int, default=32)
    parser.add_argument(
        "--host", type=str, default=os.getenv("RETRIEVAL_RERANK_HOST", DEFAULT_HOST)
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("RETRIEVAL_RERANK_PORT", str(DEFAULT_PORT))),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        RetrievalRerankConfig(
            retriever=DenseRetrieverConfig(
                model_path=args.retriever_model,
                index_path=args.index_path,
                corpus_path=args.corpus_path,
                retrieval_method=args.retrieval_method,
                topk=args.retrieval_topk,
                max_length=args.retrieval_max_length,
                use_fp16=args.retrieval_use_fp16,
                pooling_method=args.retrieval_pooling_method,
            ),
            reranker=RerankerConfig(
                model_name_or_path=args.reranker_model,
                batch_size=args.reranker_batch_size,
                rerank_topk=args.rerank_topk,
            ),
        )
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
