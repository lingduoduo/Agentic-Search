"""Shared helpers for search server implementations."""

from __future__ import annotations

import logging
from typing import TypeVar

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

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
