"""FastAPI server for dense retrieval over a local FAISS index."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .retrieval import DenseRetriever, DenseRetrieverConfig

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


@dataclass(frozen=True)
class RetrievalServerConfig:
    model_path: str
    index_path: str
    corpus_path: str
    retrieval_method: str
    topk: int = 5
    max_length: int = 180
    use_fp16: bool = False
    pooling_method: str | None = None

    def to_retriever_config(self) -> DenseRetrieverConfig:
        return DenseRetrieverConfig(
            model_path=self.model_path,
            index_path=self.index_path,
            corpus_path=self.corpus_path,
            retrieval_method=self.retrieval_method,
            topk=self.topk,
            max_length=self.max_length,
            use_fp16=self.use_fp16,
            pooling_method=self.pooling_method,
        )


class QueryRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1)
    topk: int | None = None
    return_scores: bool = False


def create_app(config: RetrievalServerConfig) -> FastAPI:
    retriever = DenseRetriever(config.to_retriever_config())
    app = FastAPI(title="Dense Retrieval Server")

    @app.get("/health")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/retrieve")
    def retrieve_endpoint(request: QueryRequest) -> dict[str, list]:
        try:
            if request.return_scores:
                return {"result": retriever.retrieve(request.queries, topk=request.topk or config.topk)}

            documents = retriever.batch_search(
                request.queries,
                num=request.topk or config.topk,
                return_score=False,
            )
            return {"result": documents}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the local dense retriever server.")
    parser.add_argument("--index_path", type=str, required=True, help="Corpus index file.")
    parser.add_argument("--corpus_path", type=str, required=True, help="Local corpus file.")
    parser.add_argument("--topk", type=int, default=5, help="Number of retrieved passages per query.")
    parser.add_argument("--retrieval_method", type=str, required=True, help="Retriever family name.")
    parser.add_argument("--model_path", type=str, required=True, help="Path or HF id for the retriever model.")
    parser.add_argument("--max_length", type=int, default=180)
    parser.add_argument("--use_fp16", action="store_true", default=False)
    parser.add_argument("--pooling_method", type=str, default=None)
    parser.add_argument("--host", type=str, default=os.getenv("RETRIEVAL_SERVER_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("RETRIEVAL_SERVER_PORT", str(DEFAULT_PORT))))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        RetrievalServerConfig(
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
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
