"""SearchTool: execute a query against a chosen retrieval backend and record it.

Supports two backends the policy can choose between per search — a vector-DB
retriever and a web retriever. When the web backend is not configured, a WEB
request degrades to the vector-DB backend (logged, never crashes), matching the
degrade-don't-crash pattern used by the routing-layer constructors.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from ...context.search import SearchResult
from ..state import Retriever, SearchAgentState

logger = logging.getLogger(__name__)

RetrieveFn = Callable[[str], Awaitable[list[SearchResult]]]


class SearchTool:
    """Run one retriever call and fold the results into the agent state.

    Caches results per ``(backend, normalized query)`` for the lifetime of the
    instance (one question), so a repeated query never re-hits the backend. A
    web call that *raises* degrades to the vector-DB backend, matching the
    "web unconfigured" degradation already in place.
    """

    def __init__(
        self, vector_db_fn: RetrieveFn, web_fn: RetrieveFn | None = None
    ) -> None:
        self._vector_db_fn = vector_db_fn
        self._web_fn = web_fn
        self._cache: dict[tuple[Retriever, str], list[SearchResult]] = {}

    async def run(
        self,
        state: SearchAgentState,
        query: str,
        retriever: Retriever = Retriever.VECTOR_DB,
    ) -> list[SearchResult]:
        key = (retriever, " ".join(query.split()).casefold())
        if key in self._cache:
            docs = self._cache[key]
        else:
            docs = await self._retrieve(retriever, query)
            self._cache[key] = docs
        state.record_search(query, docs)
        return docs

    async def _retrieve(self, retriever: Retriever, query: str) -> list[SearchResult]:
        if retriever is Retriever.WEB and self._web_fn is not None:
            try:
                return await self._web_fn(query)
            except Exception:  # noqa: BLE001 - degrade-don't-crash on web failure
                logger.warning(
                    "Web retriever raised; degrading to vector-DB backend.",
                    exc_info=True,
                )
                return await self._vector_db_fn(query)
        if retriever is Retriever.WEB:
            logger.warning(
                "Web retriever requested but not configured; "
                "degrading to vector-DB backend."
            )
        return await self._vector_db_fn(query)
