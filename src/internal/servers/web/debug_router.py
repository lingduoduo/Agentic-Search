"""Dev-console debug router — read-only observability for backend servers.

Mounted only when DEBUG_PANELS is enabled (see create_web_app). Proxies the
retrieval server's per-mode /internal/search/* endpoints so the browser can
inspect sparse/dense/hybrid/graph results without cross-origin calls.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

from src.internal.retrieval.query_transform_factory import (
    build_query_transform_pipeline_from_env,
)

_MODES = {"sparse", "dense", "hybrid", "graph"}


class DebugRetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=10, le=200)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    over_fetch: int = Field(default=2, ge=1, le=4)
    rerank: bool = False


class DebugQueryTransformRequest(BaseModel):
    query: str = Field(..., min_length=1)
    filters: dict | None = None


def _retrieval_base(search_url: str) -> str:
    """Strip a trailing /retrieve so we can address /internal/search/*."""
    return search_url.rstrip("/").removesuffix("/retrieve").rstrip("/")


def create_debug_router(
    *,
    search_url: str,
    http_client: httpx.Client | None = None,
    llm: object | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/debug", tags=["debug"])
    base = _retrieval_base(search_url)
    client = http_client or httpx.Client(timeout=15.0)

    @router.post("/query-transform")
    def query_transform(body: DebugQueryTransformRequest) -> dict:
        """Run *only* the query-transform pipeline (no retrieval).

        Returns the variants + merged filters + per-leg breakdown. When no
        pipeline is configured (no LLM / no QT_* flags) the factory returns None;
        we report ``active=false`` with the original query unchanged — never 500.
        """
        pipe = build_query_transform_pipeline_from_env(llm)
        if pipe is None:
            return {
                "original": body.query,
                "variants": [body.query],
                "merged_filters": {},
                "active": False,
                "legs": {},
            }
        bundle = pipe.transform(body.query, body.filters)
        return {
            "original": bundle.original,
            "variants": bundle.retrieval_variants(),
            "merged_filters": bundle.merged_filters,
            "active": True,
            "legs": {
                "sub_queries": bundle.sub_queries,
                "multi_query": bundle.multi_query,
                "rewrite": bundle.rewrite,
                "hyde_text": bundle.hyde_text,
                "step_back": bundle.step_back,
                "keywords": bundle.keywords,
            },
        }

    @router.get("/health")
    def health() -> dict:
        """Reachability of the configured servers. Never raises — up/down each."""
        servers = [{"name": "web", "url": "self", "status": "up"}]
        try:
            r = client.get(f"{base}/health", timeout=3.0)
            status = "up" if r.status_code == 200 else "down"
        except Exception:
            status = "down"
        servers.append({"name": "retrieval", "url": base, "status": status})
        return {"servers": servers}

    @router.post("/retrieval/{mode}")
    def retrieval(mode: str, body: DebugRetrievalRequest) -> Response:
        if mode not in _MODES:
            return Response(
                content=f'{{"detail":"unknown mode {mode!r}"}}',
                status_code=404,
                media_type="application/json",
            )
        payload: dict = {
            "query": body.query,
            "top_k": body.top_k,
            "rerank": body.rerank,
        }
        if mode == "hybrid":
            payload.update(
                rrf_k=body.rrf_k,
                mmr_lambda=body.mmr_lambda,
                over_fetch=body.over_fetch,
            )
        upstream = client.post(f"{base}/internal/search/{mode}", json=payload)
        # Pass the upstream status through verbatim — a down/misconfigured
        # retrieval server must not be masked as a generic 200/500.
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type="application/json",
        )

    return router
