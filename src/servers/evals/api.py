"""Evaluation API router.

The sampled Onyx version dispatched an eval run to Celery
(OnyxCeleryTask.EVAL_RUN_TASK) and required a cloud superuser token.

This repo has no Celery worker. The eval endpoint runs
``run_expanded_search`` synchronously and returns structured results,
making it useful for local quality measurement without any task queue.
Admin access (AppSettings.auth.super_users) is required.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from pydantic import BaseModel
from pydantic import Field

from src.auth import AuthenticatedUser
from src.auth import user_from_headers
from src.configs import AppSettings
from src.context.models import SearchFilters
from src.search.process_search_query import run_expanded_search

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class EvalConfigurationOptions(BaseModel):
    """Parameters for a single evaluation run."""

    query: str = Field(..., min_length=1)
    num_hits: int = Field(default=5, ge=1, le=50)
    run_query_expansion: bool = False
    filters: SearchFilters | None = None


class EvalResultDoc(BaseModel):
    title: str | None
    url: str | None
    score: float
    content_preview: str


class EvalRunResult(BaseModel):
    success: bool
    query: str
    executed_queries: list[str]
    num_results: int
    results: list[EvalResultDoc]
    error: str | None = None


class EvalRunAck(BaseModel):
    """Returned immediately when an eval is queued (Celery mode) or completed."""

    success: bool
    message: str | None = None


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_evals_router(
    app_settings: AppSettings,
    *,
    search_url: str = "http://localhost:8000/retrieve",
) -> APIRouter:
    """Return an APIRouter for evaluation endpoints bound to *app_settings*."""

    router = APIRouter(prefix="/evals", tags=["evals"])

    def _require_admin(request: Request) -> AuthenticatedUser:
        user = user_from_headers(request.headers)
        if user is None or user.is_anonymous:
            raise HTTPException(status_code=401, detail="Authentication required.")
        super_users = app_settings.auth.super_users
        if user.id not in super_users and (
            user.email is None or user.email not in super_users
        ):
            raise HTTPException(status_code=403, detail="Admin access required.")
        return user

    @router.post("/eval_run", response_model=EvalRunResult)
    async def eval_run(
        request: EvalConfigurationOptions,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> EvalRunResult:
        """Run a search evaluation synchronously and return structured results.

        Unlike the sampled Onyx version (Celery async), this endpoint blocks
        until the search completes and returns results directly — suitable for
        local quality measurement and CI pipelines.
        """
        logger.info("Eval run: query=%r num_hits=%d", request.query, request.num_hits)
        try:
            result = await run_expanded_search(
                request.query,
                search_url=search_url,
                top_k=request.num_hits,
                filters=request.filters,
                expand=request.run_query_expansion,
            )
            return EvalRunResult(
                success=True,
                query=request.query,
                executed_queries=result.executed_queries,
                num_results=len(result.results),
                results=[
                    EvalResultDoc(
                        title=r.title,
                        url=r.url,
                        score=r.score,
                        content_preview=r.contents[:200],
                    )
                    for r in result.results
                ],
            )
        except Exception as exc:
            logger.exception("Eval run failed for query %r", request.query)
            return EvalRunResult(
                success=False,
                query=request.query,
                executed_queries=[request.query],
                num_results=0,
                results=[],
                error=str(exc),
            )

    @router.post("/eval_run_ack")
    async def eval_run_ack(
        request: EvalConfigurationOptions,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> EvalRunAck:
        """Fire-and-forget variant: queues the eval and returns immediately.

        The search runs in a background task so the response is instant.
        Check server logs for results.
        """

        async def _run_background() -> None:
            try:
                result = await run_expanded_search(
                    request.query,
                    search_url=search_url,
                    top_k=request.num_hits,
                    filters=request.filters,
                    expand=request.run_query_expansion,
                )
                logger.info(
                    "Background eval complete: query=%r hits=%d",
                    request.query,
                    len(result.results),
                )
            except Exception as exc:
                logger.error("Background eval failed: %s", exc)

        asyncio.ensure_future(_run_background())
        return EvalRunAck(success=True, message="Eval queued.")

    return router


__all__ = [
    "EvalConfigurationOptions",
    "EvalRunAck",
    "EvalRunResult",
    "create_evals_router",
]
