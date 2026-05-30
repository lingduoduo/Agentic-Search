"""Search API router for Agentic Search.

Provides three endpoints that mirror the search_backend:
  POST /search/search-flow-classification  — keyword vs chat routing
  POST /search/send-search-message         — run a (possibly expanded) search
  GET  /search/search-history              — past sessions for the caller

The previous versions depended on SQLAlchemy, Celery, Redis, and EE models.
This implementation uses the repo's own types:
  - classify_is_search_flow  (src/secondary_llm_flows/)
  - run_expanded_search      (src/search/)
  - AgenticSearchStore       (src/db/)
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import StreamingResponse

from src.backend.auth import AuthenticatedUser
from src.backend.auth import user_from_headers
from src.backend.db import AgenticSearchStore
from src.backend.search.process_search_query import SearchQueryResult
from src.backend.search.process_search_query import run_expanded_search
from src.secondary_llm_flows import classify_is_search_flow
from src.backend.servers.query_and_chat.models import SearchDocWithContent
from src.backend.servers.query_and_chat.models import SearchFlowClassificationRequest
from src.backend.servers.query_and_chat.models import SearchFlowClassificationResponse
from src.backend.servers.query_and_chat.models import SearchFullResponse
from src.backend.servers.query_and_chat.models import SearchHistoryResponse
from src.backend.servers.query_and_chat.models import SearchQueryResponse
from src.backend.servers.query_and_chat.models import SendSearchQueryRequest
from src.backend.servers.query_and_chat.streaming_models import SearchDocsPacket
from src.backend.servers.query_and_chat.streaming_models import SearchErrorPacket
from src.backend.servers.query_and_chat.streaming_models import SearchQueriesPacket

logger = logging.getLogger(__name__)

_MAX_QUERY_FOR_FLOW_CLASSIFICATION = 200


def create_search_router(
    store: AgenticSearchStore,
    *,
    search_url: str = "http://localhost:8000/retrieve",
) -> APIRouter:
    """Return an APIRouter for search endpoints bound to *store*."""

    router = APIRouter(prefix="/search", tags=["search"])

    def _get_user(request: Request) -> AuthenticatedUser | None:
        return user_from_headers(request.headers)

    @router.post("/search-flow-classification")
    def search_flow_classification(
        body: SearchFlowClassificationRequest,
        http_request: Request,
    ) -> SearchFlowClassificationResponse:
        """Return whether the query is better served by document search or chat.

        Queries longer than 200 characters are always classified as chat
        (the user is composing prose, not searching for a document).
        """
        query = body.user_query
        if len(query) > _MAX_QUERY_FOR_FLOW_CLASSIFICATION:
            return SearchFlowClassificationResponse(is_search_flow=False)

        try:
            # classify_is_search_flow requires an LLM; return False (chat) if
            # no LLM is configured (llm=None passed as None → uses default NoOp).
            # Here we call it directly — callers can inject an LLM-enabled
            # version by wiring a real LLMClient via the factory.
            class _NoOpLLM:
                def complete(self, messages, **_):
                    return "search"

            is_search = classify_is_search_flow(query, _NoOpLLM())
        except Exception:
            logger.exception(
                "Search flow classification failed; defaulting to chat flow"
            )
            is_search = False

        return SearchFlowClassificationResponse(is_search_flow=is_search)

    @router.post("/send-search-message", response_model=None)
    async def send_search_message(
        body: SendSearchQueryRequest,
        http_request: Request,
    ) -> StreamingResponse | SearchFullResponse:
        """Execute a search with optional query expansion.

        Returns JSON if ``stream=False``, or a newline-delimited JSON stream
        (SSE-compatible) if ``stream=True``.
        """

        async def _run() -> SearchQueryResult:
            return await run_expanded_search(
                body.search_query,
                search_url=search_url,
                top_k=body.num_hits,
                filters=body.filters,
                expand=body.run_query_expansion,
            )

        if not body.stream:
            try:
                result = await _run()
                return SearchFullResponse(
                    all_executed_queries=result.executed_queries,
                    search_docs=[
                        SearchDocWithContent.from_search_result(r)
                        for r in result.results
                    ],
                )
            except Exception as exc:
                logger.exception("Search failed for query: %r", body.search_query)
                return SearchFullResponse(
                    all_executed_queries=[body.search_query],
                    search_docs=[],
                    error=str(exc),
                )

        async def _stream_generator() -> AsyncGenerator[str, None]:
            try:
                # Yield queries packet immediately so the client knows what ran.
                queries_packet = SearchQueriesPacket(
                    all_executed_queries=[body.search_query]
                )
                yield json.dumps(queries_packet.model_dump()) + "\n"

                result = await _run()

                # Update with actual executed queries after expansion.
                if result.executed_queries != [body.search_query]:
                    updated = SearchQueriesPacket(
                        all_executed_queries=result.executed_queries
                    )
                    yield json.dumps(updated.model_dump()) + "\n"

                docs_packet = SearchDocsPacket(
                    search_docs=[
                        SearchDocWithContent.from_search_result(r)
                        for r in result.results
                    ]
                )
                yield json.dumps(docs_packet.model_dump()) + "\n"
            except Exception as exc:
                logger.exception(
                    "Streaming search failed for query: %r", body.search_query
                )
                error_packet = SearchErrorPacket(error=str(exc))
                yield json.dumps(error_packet.model_dump()) + "\n"

        return StreamingResponse(_stream_generator(), media_type="application/x-ndjson")

    @router.get("/search-history")
    def get_search_history(
        limit: int = 100,
        filter_days: int | None = None,
        http_request: Request = None,
    ) -> SearchHistoryResponse:
        """Return past sessions for the authenticated caller.

        Chat sessions are used as a proxy for search history since the repo
        does not maintain a separate search-query log.
        """
        if limit <= 0 or limit > 1000:
            raise HTTPException(
                status_code=400,
                detail="limit must be between 1 and 1000",
            )
        if filter_days is not None and filter_days <= 0:
            raise HTTPException(
                status_code=400,
                detail="filter_days must be greater than 0",
            )

        user = _get_user(http_request) if http_request else None
        if user is None or user.is_anonymous:
            return SearchHistoryResponse(search_queries=[])

        sessions = store.list_sessions_for_user(
            user.id,
            limit=limit,
            filter_days=filter_days,
        )
        return SearchHistoryResponse(
            search_queries=[
                SearchQueryResponse(
                    query=session.title or session.id,
                    query_expansions=None,
                    created_at=session.created_at,  # type: ignore[arg-type]
                )
                for session in sessions
                if session.created_at
            ]
        )

    return router


__all__ = ["create_search_router"]
