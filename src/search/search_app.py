"""Shared helpers for search server implementations."""

from __future__ import annotations

from typing import TypeVar

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

T = TypeVar("T")


class SearchRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1)


def format_document(title: str | None, content: str | None) -> dict[str, dict[str, str]]:
    normalized_title = title or "No title."
    normalized_content = content or "No snippet available."
    return {"document": {"contents": f"\"{normalized_title}\"\n{normalized_content}"}}


def create_search_app(title: str, engine: T) -> FastAPI:
    app = FastAPI(title=title)

    @app.get("/health")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/retrieve")
    def search_endpoint(request: SearchRequest) -> dict[str, list]:
        try:
            results = engine.batch_search(request.queries)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"result": results}

    return app
