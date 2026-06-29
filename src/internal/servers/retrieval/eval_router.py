"""Internal eval endpoints: /internal/search/{sparse,dense,hybrid}.

Pass require_admin=make_require_admin(app_settings) in production.
Pass require_admin=None in tests to skip auth.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.fusion import mmr_rerank, rrf_fuse
from src.internal.retrieval.reranker_factory import build_reranker_from_env
from src.internal.retrieval.service import RetrievalService
from src.internal.servers.retrieval.server import SearchResponse, _to_item


def _maybe_rerank(
    results: list[RetrievalResult], query: str, top_k: int, mode: str, do_rerank: bool
) -> tuple[list[RetrievalResult], str]:
    """Apply the env-configured cross-encoder when requested.

    Returns ``(reranked, "{mode}+reranked")`` if a reranker is configured, else
    ``(results, mode)`` unchanged — so "no reranker active" stays visible and the
    debug surface never errors.
    """
    if not do_rerank:
        return results, mode
    reranker = build_reranker_from_env()
    if reranker is None:
        return results, mode
    return reranker.rerank(query, results, top_k), f"{mode}+reranked"


class InternalSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    rerank: bool = False


class HybridSearchRequest(InternalSearchRequest):
    rrf_k: int = Field(default=60, ge=10, le=200)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    over_fetch: int = Field(default=2, ge=1, le=4)


class GraphSearchRequest(InternalSearchRequest):
    initial_k: int = Field(default=5, ge=1, le=50)
    max_entity_queries: int = Field(default=3, ge=0, le=10)


def create_eval_router(
    service: RetrievalService,
    require_admin: Callable | None = None,
) -> APIRouter:
    """Return router with /internal/search/{sparse,dense,hybrid} endpoints."""
    router = APIRouter(prefix="/internal/search")
    deps = [Depends(require_admin)] if require_admin is not None else []

    @router.post("/sparse", response_model=SearchResponse, dependencies=deps)
    def search_sparse(request: InternalSearchRequest) -> SearchResponse:
        t0 = time.monotonic()
        results = service._backend.search_sparse(request.query, top_k=request.top_k)
        results, mode = _maybe_rerank(
            results, request.query, request.top_k, "sparse", request.rerank
        )
        return SearchResponse(
            results=[_to_item(r) for r in results],
            retrieval_mode=mode,
            executed_queries=[request.query],
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    @router.post("/dense", response_model=SearchResponse, dependencies=deps)
    def search_dense(request: InternalSearchRequest) -> SearchResponse:
        t0 = time.monotonic()
        try:
            results = service._backend.search_dense(request.query, top_k=request.top_k)
        except NotImplementedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        results, mode = _maybe_rerank(
            results, request.query, request.top_k, "dense", request.rerank
        )
        return SearchResponse(
            results=[_to_item(r) for r in results],
            retrieval_mode=mode,
            executed_queries=[request.query],
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    @router.post("/hybrid", response_model=SearchResponse, dependencies=deps)
    def search_hybrid(request: HybridSearchRequest) -> SearchResponse:
        t0 = time.monotonic()
        over_fetch = request.top_k * request.over_fetch
        sparse = service._backend.search_sparse(request.query, top_k=over_fetch)
        try:
            dense: list[RetrievalResult] = service._backend.search_dense(
                request.query, top_k=over_fetch
            )
        except NotImplementedError:
            dense = []
        fused = rrf_fuse([sparse, dense] if dense else [sparse], rrf_k=request.rrf_k)
        mmr = mmr_rerank(fused, top_k=request.top_k, mmr_lambda=request.mmr_lambda)
        results, mode = _maybe_rerank(
            mmr, request.query, request.top_k, "hybrid", request.rerank
        )
        return SearchResponse(
            results=[_to_item(r) for r in results],
            retrieval_mode=mode,
            executed_queries=[request.query],
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    @router.post("/graph", response_model=SearchResponse, dependencies=deps)
    def search_graph(request: GraphSearchRequest) -> SearchResponse:
        t0 = time.monotonic()
        results = service.graph_search(
            request.query,
            top_k=request.top_k,
            initial_k=request.initial_k,
            max_entity_queries=request.max_entity_queries,
        )
        results, mode = _maybe_rerank(
            results, request.query, request.top_k, "graph", request.rerank
        )
        return SearchResponse(
            results=[_to_item(r) for r in results],
            retrieval_mode=mode,
            executed_queries=[request.query],
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    return router
