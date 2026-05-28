"""Shared helpers for search server implementations."""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import TypeVar

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080

T = TypeVar("T")


class SearchRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1)


class FetchRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1)


def format_document(
    title: str | None,
    content: str | None,
    url: str | None = None,
) -> dict[str, dict[str, str]]:
    normalized_title = title or "No title."
    normalized_content = content or "No snippet available."
    document = {
        "title": normalized_title,
        "contents": f'"{normalized_title}"\n{normalized_content}',
    }
    if url:
        document["url"] = url
    return {"document": document}


def create_base_app(title: str) -> FastAPI:
    app = FastAPI(title=title)

    @app.get("/health")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    return app


def create_search_app(title: str, engine: T) -> FastAPI:
    app = create_base_app(title)

    @app.post("/retrieve")
    def search_endpoint(request: SearchRequest) -> dict[str, list]:
        try:
            results = engine.batch_search(request.queries)
        except Exception as exc:
            logger.exception("Search request failed: %s", exc)
            raise HTTPException(
                status_code=500, detail="Search request failed"
            ) from exc
        return {"result": results}

    if hasattr(engine, "fetch_urls"):

        @app.post("/fetch")
        def fetch_endpoint(request: FetchRequest) -> dict[str, list]:
            try:
                results = engine.fetch_urls(request.urls)
            except Exception as exc:
                logger.exception("Fetch request failed: %s", exc)
                raise HTTPException(
                    status_code=500, detail="Fetch request failed"
                ) from exc
            return {"result": results}

    return app


def add_host_port_args(
    parser: argparse.ArgumentParser,
    host_env: str,
    port_env: str,
    default_host: str = DEFAULT_HOST,
    default_port: int = DEFAULT_PORT,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv(host_env, default_host),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv(port_env, str(default_port))),
    )
    return parser


def load_environment(override: bool = True) -> None:
    from dotenv import load_dotenv

    load_dotenv(override=override)


def run_uvicorn_app(app: FastAPI, host: str, port: int, workers: int = 1) -> None:
    uvicorn.run(app, host=host, port=port, workers=workers)
