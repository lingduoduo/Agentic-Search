"""FastAPI server for reranking candidate documents."""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn
from fastapi import HTTPException
from pydantic import BaseModel, Field

from .rerank import RerankerConfig, get_reranker, passage_to_string
from .search_app import create_base_app

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 6980


class RerankRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1)
    documents: list[list[dict]]
    rerank_topk: int | None = None
    return_scores: bool = False


def create_app(config: RerankerConfig):
    reranker = get_reranker(config)
    app = create_base_app("Rerank Server")

    @app.post("/rerank")
    def rerank_endpoint(request: RerankRequest) -> dict[str, list]:
        try:
            reranked = reranker.rerank(
                request.queries,
                request.documents,
                topk=request.rerank_topk or config.rerank_topk,
            )
        except Exception as exc:
            logger.exception("Rerank request failed: %s", exc)
            raise HTTPException(
                status_code=500, detail="Rerank request failed"
            ) from exc

        if request.return_scores:
            return {"result": reranked}

        return {
            "result": [
                [passage_to_string(item["document"]) for item in query_results]
                for query_results in reranked
            ]
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the rerank server.")
    parser.add_argument(
        "--rerank_model_name_or_path",
        type=str,
        default=os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L12-v2"),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--rerank_topk", type=int, default=3)
    parser.add_argument(
        "--host", type=str, default=os.getenv("RERANK_SERVER_HOST", DEFAULT_HOST)
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("RERANK_SERVER_PORT", str(DEFAULT_PORT))),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        RerankerConfig(
            model_name_or_path=args.rerank_model_name_or_path,
            batch_size=args.batch_size,
            rerank_topk=args.rerank_topk,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
