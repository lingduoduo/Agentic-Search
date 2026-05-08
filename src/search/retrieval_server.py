"""FastAPI server for dense retrieval over a local FAISS index."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass

import uvicorn
from fastapi import HTTPException
from pydantic import BaseModel, Field

from .retrieval import DenseRetriever, DenseRetrieverConfig
from .search_app import create_base_app

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


@dataclass(frozen=True)
class RetrievalServerConfig:
    """Runtime configuration for the standalone retrieval service."""

    retriever: DenseRetrieverConfig
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT


class RetrieveRequest(BaseModel):
    query: str | None = None
    queries: list[str] | None = Field(default=None, min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    topk: int | None = Field(default=None, ge=1)
    return_scores: bool = False

    def resolved_queries(self) -> list[str]:
        if self.query is not None:
            return [self.query]
        if self.queries:
            return list(self.queries)
        raise ValueError("Either `query` or `queries` must be provided.")

    def resolved_topk(self, default_topk: int) -> int:
        return self.top_k or self.topk or default_topk

    def is_single_query(self) -> bool:
        return self.query is not None


def _flatten_document_row(row: dict) -> dict[str, object]:
    document = row.get("document", row)
    if not isinstance(document, dict):
        document = {}
    text = document.get("text")
    if not isinstance(text, str):
        text = document.get("contents", "")
    return {
        "doc_id": document.get("id") or document.get("doc_id"),
        "score": row.get("score"),
        "title": document.get("title"),
        "text": text,
        "url": document.get("url"),
    }


def create_app(config: DenseRetrieverConfig | RetrievalServerConfig):
    server_config = (
        config
        if isinstance(config, RetrievalServerConfig)
        else RetrievalServerConfig(config)
    )
    retriever = DenseRetriever(server_config.retriever)
    app = create_base_app("Dense Retrieval Server")

    @app.post("/retrieve")
    def retrieve_endpoint(request: RetrieveRequest) -> dict[str, object]:
        try:
            queries = request.resolved_queries()
            topk = request.resolved_topk(server_config.retriever.topk)
            if request.return_scores:
                rows = retriever.retrieve(queries, topk=topk)
            else:
                documents = retriever.batch_search(
                    queries,
                    num=topk,
                    return_score=False,
                )
                rows = documents

            if request.is_single_query():
                single_rows = rows[0] if rows else []
                results = [
                    _flatten_document_row(row) if isinstance(row, dict) else {}
                    for row in single_rows
                ]
                return {
                    "query": queries[0],
                    "top_k": topk,
                    "results": results,
                    # legacy field for existing clients
                    "result": [single_rows],
                }

            return {
                "queries": queries,
                "top_k": topk,
                "results": rows,
                # legacy field for existing clients
                "result": rows,
            }
        except Exception as exc:
            logger.exception("Retrieval request failed: %s", exc)
            raise HTTPException(
                status_code=500, detail="Retrieval request failed"
            ) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the local dense retriever server."
    )
    parser.add_argument(
        "--index_path", type=str, required=True, help="Corpus index file."
    )
    parser.add_argument(
        "--corpus_path", type=str, required=True, help="Local corpus file."
    )
    parser.add_argument(
        "--topk", type=int, default=5, help="Number of retrieved passages per query."
    )
    parser.add_argument(
        "--retrieval_method", type=str, required=True, help="Retriever family name."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path or HF id for the retriever model.",
    )
    parser.add_argument("--max_length", type=int, default=180)
    parser.add_argument("--use_fp16", action="store_true", default=False)
    parser.add_argument("--pooling_method", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Device for the embedding model and FAISS index.  Defaults to 'cpu' "
            "so the retrieval service does not compete with the trainer for GPU "
            "memory.  Pass 'cuda' or 'cuda:1' only when the retrieval server runs "
            "on a dedicated GPU."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes (production multi-process serving).",
    )
    parser.add_argument(
        "--host", type=str, default=os.getenv("RETRIEVAL_SERVER_HOST", DEFAULT_HOST)
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("RETRIEVAL_SERVER_PORT", str(DEFAULT_PORT))),
    )
    return parser.parse_args()


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(override=True)
    args = parse_args()
    app = create_app(
        RetrievalServerConfig(
            retriever=DenseRetrieverConfig(
                model_path=args.model_path,
                index_path=args.index_path,
                corpus_path=args.corpus_path,
                retrieval_method=args.retrieval_method,
                topk=args.topk,
                max_length=args.max_length,
                use_fp16=args.use_fp16,
                pooling_method=args.pooling_method,
                device=args.device,
            ),
            host=args.host,
            port=args.port,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
