"""POST /api/feedback — persists thumbs_up / thumbs_down signals."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from src.internal.db import AgenticSearchStore


class FeedbackRequest(BaseModel):
    session_id: str
    signal: Literal["thumbs_up", "thumbs_down"]
    note: str | None = None
    source: str | None = None
    parent_feedback_id: str | None = None
    correlation_id: str | None = None


class FeedbackResponse(BaseModel):
    ok: bool


def create_feedback_router(db: AgenticSearchStore) -> APIRouter:
    """Return a router with POST /api/feedback."""
    router = APIRouter(tags=["feedback"])

    @router.post("/api/feedback", response_model=FeedbackResponse)
    def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
        db.save_retrieval_feedback(
            request.session_id,
            request.signal,
            note=request.note,
            source=request.source,
            parent_feedback_id=request.parent_feedback_id,
            correlation_id=request.correlation_id,
        )
        return FeedbackResponse(ok=True)

    return router
