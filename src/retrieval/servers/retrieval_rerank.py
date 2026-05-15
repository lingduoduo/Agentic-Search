"""FastAPI server that combines local retrieval with cross-encoder reranking."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass

import uvicorn
from fastapi import HTTPException
from pydantic import BaseModel, Field

from .. import build_retriever
from ..dense_retriever import DenseRetrieverConfig
from ..sparse_retriever import SparseRetrieverConfig
from ..rerank import RerankerConfig, get_reranker
from .app import create_base_app

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


@dataclass(frozen=True)
class RetrievalRerankConfig:
    retriever: DenseRetrieverConfig | SparseRetrieverConfig
    reranker: RerankerConfig


class SearchRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1)
    topk_retrieval: int | None = Field(default=None, ge=1)
    topk_rerank: int | None = Field(default=None, ge=1)
    # Kept for request compatibility. Retrieval+rerank always returns
    # cross-encoder scores because they are the output of the second stage.
    return_scores: bool = False


def create_app(config: RetrievalRerankConfig):
    retriever = build_retriever(config.retriever)
    reranker = get_reranker(config.reranker)
    app = create_base_app("Retrieval and Rerank Server")

    @app.post("/retrieve")
    def search_endpoint(request: SearchRequest) -> dict[str, list]:
        try:
            topk_retrieval = (
                request.topk_retrieval
                if request.topk_retrieval is not None
                else config.retriever.topk
            )
            retrieved = retriever.batch_search(
                request.queries,
                num=topk_retrieval,
                return_score=False,
            )
            reranked = reranker.rerank(
                request.queries,
                retrieved,
                topk=(
                    request.topk_rerank
                    if request.topk_rerank is not None
                    else config.reranker.rerank_topk
                ),
            )
        except Exception as exc:
            logger.exception("Retrieval+rerank request failed: %s", exc)
            raise HTTPException(
                status_code=500, detail="Retrieval+rerank request failed"
            ) from exc

        return {"result": reranked}

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
    parser.add_argument(
        "--retriever_model",
        type=str,
        default=None,
        help="Path or HF id for dense retrieval methods. Not required for BM25.",
    )
    parser.add_argument("--retrieval_max_length", type=int, default=180)
    parser.add_argument("--retrieval_query_batch_size", type=int, default=128)
    parser.add_argument("--retrieval_use_fp16", action="store_true", default=False)
    parser.add_argument("--retrieval_pooling_method", type=str, default=None)
    parser.add_argument("--faiss_gpu", action="store_true", default=False)
    parser.add_argument(
        "--hnsw_ef_search",
        type=int,
        default=None,
        help="Optional FAISS HNSW efSearch value for CPU ANN retrieval.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Device for the dense embedding model. Defaults to 'cpu' so the "
            "retrieval service does not compete with trainer GPU memory."
        ),
    )
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
    retrieval_method = args.retrieval_method.lower()
    if retrieval_method == "bm25":
        retriever_config = SparseRetrieverConfig(
            index_path=args.index_path,
            corpus_path=args.corpus_path,
            retrieval_method=retrieval_method,
            topk=args.retrieval_topk,
        )
    else:
        if not args.retriever_model:
            raise ValueError(
                "--retriever_model is required for dense retrieval methods."
            )
        retriever_config = DenseRetrieverConfig(
            model_path=args.retriever_model,
            index_path=args.index_path,
            corpus_path=args.corpus_path,
            retrieval_method=args.retrieval_method,
            topk=args.retrieval_topk,
            max_length=args.retrieval_max_length,
            query_batch_size=args.retrieval_query_batch_size,
            use_fp16=args.retrieval_use_fp16,
            pooling_method=args.retrieval_pooling_method,
            faiss_gpu=args.faiss_gpu,
            hnsw_ef_search=args.hnsw_ef_search,
            device=args.device,
        )
    app = create_app(
        RetrievalRerankConfig(
            retriever=retriever_config,
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
