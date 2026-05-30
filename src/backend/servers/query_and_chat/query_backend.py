"""Query API router — stub for EE-only features.

Standard Answers (the query_backend.py) are an Enterprise
feature that requires a separate Slack/bot integration and DB tables
not present in this repo. The endpoint is preserved as a documented stub
that returns HTTP 501 so API clients that probe for the feature receive
a clear signal rather than a 404.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi import HTTPException

basic_router = APIRouter(prefix="/query", tags=["query"])


@basic_router.get("/standard-answer")
def get_standard_answer() -> None:
    """Standard Answers is an Enterprise feature not available in this deployment."""
    raise HTTPException(
        status_code=501,
        detail=(
            "Standard Answers is an Enterprise feature and is not available "
            "in this deployment."
        ),
    )


__all__ = ["basic_router"]
