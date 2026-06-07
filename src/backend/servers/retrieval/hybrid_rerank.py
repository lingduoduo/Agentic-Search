"""FastAPI server combining hybrid retrieval (dense+sparse RRF) with cross-encoder reranking."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.backend.document_index.retrieval import (
    DenseRetrieverConfig,
    SparseRetrieverConfig,
)
from src.backend.document_index.hybrid_retriever import (
    HybridRetriever,
    HybridRetrieverConfig,
)
from src.backend.servers.retrieval.rerank import RerankerConfig, get_reranker

from .app import add_host_port_args, create_base_app, load_environment, run_uvicorn_app

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


@dataclass(frozen=True)
class HybridRerankConfig:
    retriever: HybridRetrieverConfig
    reranker: RerankerConfig


class SearchRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1)
    topk_retrieval: int | None = Field(default=None, ge=1)
    topk_rerank: int | None = Field(default=None, ge=1)
    return_scores: bool = False


def create_app(config: HybridRerankConfig):
    retriever = HybridRetriever(config.retriever)
    reranker = get_reranker(config.reranker)
    app = create_base_app("Hybrid Retrieval + Rerank Server")

    @app.post("/retrieve")
    def search_endpoint(request: SearchRequest) -> dict[str, list]:
        try:
            topk_retrieval = request.topk_retrieval or config.retriever.dense.topk
            topk_rerank = request.topk_rerank or config.reranker.rerank_topk
            doc_lists = retriever.batch_search(
                request.queries, num=topk_retrieval, return_score=False
            )
            reranked = reranker.rerank(request.queries, doc_lists, topk=topk_rerank)
        except Exception as exc:
            logger.exception("Hybrid+rerank request failed: %s", exc)
            raise HTTPException(
                status_code=500, detail="Hybrid+rerank request failed"
            ) from exc
        return {"result": reranked}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the hybrid retrieval + rerank server."
    )
    parser.add_argument(
        "--dense_model",
        type=str,
        required=True,
        help="Dense encoder model path or HF id.",
    )
    parser.add_argument(
        "--index_path", type=str, required=True, help="FAISS index path."
    )
    parser.add_argument(
        "--corpus_path", type=str, required=True, help="Corpus JSONL path."
    )
    parser.add_argument("--retrieval_method", type=str, default="e5")
    parser.add_argument("--retrieval_topk", type=int, default=10)
    parser.add_argument(
        "--sparse_index_path",
        type=str,
        default=None,
        help="BM25 index path (enables hybrid mode).",
    )
    parser.add_argument(
        "--hybrid_alpha",
        type=float,
        default=1.0,
        help="1.0=pure dense, 0.0=pure BM25, in-between=hybrid.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--reranker_model", type=str, default="cross-encoder/ms-marco-MiniLM-L12-v2"
    )
    parser.add_argument("--rerank_topk", type=int, default=5)
    parser.add_argument("--reranker_batch_size", type=int, default=32)
    add_host_port_args(
        parser, "HYBRID_RERANK_HOST", "HYBRID_RERANK_PORT", DEFAULT_HOST, DEFAULT_PORT
    )
    return parser.parse_args()


def main() -> None:
    load_environment()
    args = parse_args()

    sparse_config = None
    if args.sparse_index_path:
        sparse_config = SparseRetrieverConfig(
            index_path=args.sparse_index_path,
            corpus_path=args.corpus_path,
            retrieval_method="bm25",
            topk=args.retrieval_topk,
        )

    config = HybridRerankConfig(
        retriever=HybridRetrieverConfig(
            dense=DenseRetrieverConfig(
                model_path=args.dense_model,
                index_path=args.index_path,
                corpus_path=args.corpus_path,
                retrieval_method=args.retrieval_method,
                topk=args.retrieval_topk,
                device=args.device,
            ),
            sparse=sparse_config,
            hybrid_alpha=args.hybrid_alpha,
        ),
        reranker=RerankerConfig(
            model_name_or_path=args.reranker_model,
            batch_size=args.reranker_batch_size,
            rerank_topk=args.rerank_topk,
        ),
    )
    app = create_app(config)
    run_uvicorn_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
