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
    """Run one retriever call and fold the results into the agent state."""

    def __init__(
        self, vector_db_fn: RetrieveFn, web_fn: RetrieveFn | None = None
    ) -> None:
        self._vector_db_fn = vector_db_fn
        self._web_fn = web_fn

    async def run(
        self,
        state: SearchAgentState,
        query: str,
        retriever: Retriever = Retriever.VECTOR_DB,
    ) -> list[SearchResult]:
        docs = await self._select(retriever)(query)
        state.record_search(query, docs)
        return docs

    def _select(self, retriever: Retriever) -> RetrieveFn:
        if retriever is Retriever.WEB:
            if self._web_fn is None:
                logger.warning(
                    "Web retriever requested but not configured; "
                    "degrading to vector-DB backend."
                )
                return self._vector_db_fn
            return self._web_fn
        return self._vector_db_fn
