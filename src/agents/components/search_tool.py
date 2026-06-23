"""SearchTool: execute a query against a retrieval backend and record it.

In Phase A this wraps a single injected retrieval callable (behavior-preserving).
Phase B (T-B.2) extends it to select between a web and a vector-DB backend.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from ...context.search import SearchResult
from ..state import SearchAgentState

RetrieveFn = Callable[[str], Awaitable[list[SearchResult]]]


class SearchTool:
    """Run one retriever call and fold the results into the agent state."""

    def __init__(self, retrieve_fn: RetrieveFn) -> None:
        self._retrieve_fn = retrieve_fn

    async def run(self, state: SearchAgentState, query: str) -> list[SearchResult]:
        docs = await self._retrieve_fn(query)
        state.record_search(query, docs)
        return docs
