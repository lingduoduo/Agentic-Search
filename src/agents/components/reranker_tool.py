"""RerankerTool: reorder the working document set via an injected reranker.

The rerank function maps ``(query, docs) -> reordered docs`` (most relevant
first). In Phase A the tool exists but is only invoked explicitly; Phase B
(T-B.3) wires it as a policy action backed by the cross-encoder reranker.
Reranking is not a retriever call, so it does not advance ``search_rounds``.
"""

from __future__ import annotations

from collections.abc import Callable

from ...context.search import SearchResult
from ..state import SearchAgentState

RerankFn = Callable[[str, list[SearchResult]], list[SearchResult]]


class RerankerTool:
    """Re-order ``state.retrieved_docs`` in place using the rerank function."""

    def __init__(self, rerank_fn: RerankFn) -> None:
        self._rerank_fn = rerank_fn

    def run(
        self, state: SearchAgentState, query: str | None = None
    ) -> list[SearchResult]:
        if not state.retrieved_docs:
            return []
        reordered = self._rerank_fn(query or state.question, list(state.retrieved_docs))
        state.record_rerank(reordered)
        return reordered
