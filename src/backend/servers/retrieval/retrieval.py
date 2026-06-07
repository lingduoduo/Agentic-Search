"""FastAPI server for local dense or sparse retrieval indexes."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.backend.document_index.retrieval import build_retriever
from src.backend.document_index.retrieval import (
    DenseRetriever,
    DenseRetrieverConfig,
    SparseRetriever,
    SparseRetrieverConfig,
)
from .app import (
    add_host_port_args,
    create_base_app,
    load_environment,
    run_uvicorn_app,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


@dataclass(frozen=True)
class RetrievalServerConfig:
    """Runtime configuration for the standalone retrieval service."""

    retriever: DenseRetrieverConfig | SparseRetrieverConfig
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT


class RetrieveRequest(BaseModel):
    query: str | None = None
    queries: list[str] | None = Field(default=None, min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    topk: int | None = Field(default=None, ge=1)
    return_scores: bool = False
    filters: dict[str, object] | None = None

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


def _retrieve_rows(
    retriever: DenseRetriever | SparseRetriever,
    queries: list[str],
    *,
    topk: int,
    return_scores: bool,
) -> list[list[dict]]:
    rows = retriever.retrieve(queries, topk=topk)
    if return_scores:
        return rows
    return [[item["document"] for item in row] for row in rows]


def create_app(
    config: DenseRetrieverConfig | SparseRetrieverConfig | RetrievalServerConfig,
):
    server_config = (
        config
        if isinstance(config, RetrievalServerConfig)
        else RetrievalServerConfig(config)
    )
    retriever = build_retriever(server_config.retriever)
    app = create_base_app("Local Retrieval Server")

    @app.post("/retrieve")
    def retrieve_endpoint(request: RetrieveRequest) -> dict[str, object]:
        try:
            queries = request.resolved_queries()
            topk = request.resolved_topk(server_config.retriever.topk)
            rows = _retrieve_rows(
                retriever,
                queries,
                topk=topk,
                return_scores=request.return_scores,
            )
            if request.filters:
                rows = [
                    [
                        item
                        for item in row
                        if _matches_request_filters(item, request.filters or {})
                    ]
                    for row in rows
                ]

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


def _matches_request_filters(item: dict, filters: dict[str, object]) -> bool:
    document = item.get("document", item)
    if not isinstance(document, dict):
        return False
    metadata = dict(document.get("metadata") or {})
    for key in (
        "source_type",
        "document_sets",
        "document_set",
        "tags",
        "acl",
        "updated_at",
    ):
        if key in document and key not in metadata:
            metadata[key] = document[key]

    source_types = filters.get("source_types")
    if source_types and metadata.get("source_type") not in _as_values(source_types):
        return False

    document_sets = filters.get("document_sets")
    if document_sets:
        values = _as_values(metadata.get("document_sets", metadata.get("document_set")))
        if not values.intersection(_as_values(document_sets)):
            return False

    tags = filters.get("tags")
    if isinstance(tags, dict) and tags:
        metadata_tags = metadata.get("tags", metadata)
        if not isinstance(metadata_tags, dict):
            return False
        for key, value in tags.items():
            if metadata_tags.get(key) != value:
                return False

    access_acl = filters.get("access_acl")
    if access_acl:
        acl = _as_values(metadata.get("acl"))
        metadata_tags = metadata.get("tags")
        if isinstance(metadata_tags, dict):
            acl.update(_as_values(metadata_tags.get("acl")))
        if not acl.intersection(_as_values(access_acl)):
            return False

    return True


def _as_values(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(part).strip() for part in value if str(part).strip()}
    return {str(value)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the local dense or sparse retriever server."
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
        default=None,
        help="Path or HF id for the dense retriever model. Not required for BM25.",
    )
    parser.add_argument("--max_length", type=int, default=180)
    parser.add_argument("--query_batch_size", type=int, default=128)
    parser.add_argument("--use_fp16", action="store_true", default=False)
    parser.add_argument("--pooling_method", type=str, default=None)
    parser.add_argument("--faiss_gpu", action="store_true", default=False)
    parser.add_argument(
        "--normalize_query_embeddings", action="store_true", default=False
    )
    parser.add_argument("--query_prefix", type=str, default=None)
    parser.add_argument("--passage_prefix", type=str, default=None)
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
    add_host_port_args(parser, "RETRIEVAL_SERVER_HOST", "RETRIEVAL_SERVER_PORT")
    return parser.parse_args()


def main() -> None:
    load_environment()
    args = parse_args()
    retrieval_method = args.retrieval_method.lower()
    if retrieval_method == "bm25":
        retriever_config = SparseRetrieverConfig(
            index_path=args.index_path,
            corpus_path=args.corpus_path,
            retrieval_method=retrieval_method,
            topk=args.topk,
        )
    else:
        if not args.model_path:
            raise ValueError("--model_path is required for dense retrieval methods.")
        retriever_config = DenseRetrieverConfig(
            model_path=args.model_path,
            index_path=args.index_path,
            corpus_path=args.corpus_path,
            retrieval_method=args.retrieval_method,
            topk=args.topk,
            max_length=args.max_length,
            query_batch_size=args.query_batch_size,
            use_fp16=args.use_fp16,
            pooling_method=args.pooling_method,
            device=args.device,
            faiss_gpu=args.faiss_gpu,
            normalize_query_embeddings=args.normalize_query_embeddings,
            query_prefix=args.query_prefix,
            passage_prefix=args.passage_prefix,
            hnsw_ef_search=args.hnsw_ef_search,
        )
    app = create_app(
        RetrievalServerConfig(
            retriever=retriever_config,
            host=args.host,
            port=args.port,
        )
    )
    run_uvicorn_app(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
