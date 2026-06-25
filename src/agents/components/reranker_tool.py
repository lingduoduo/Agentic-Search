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
    """Re-order ``state.retrieved_docs`` in place using the rerank function.

    ``max_candidates`` bounds cost: only the top-N docs (by current order) are
    scored by the cross-encoder; the rest keep their position. ``None`` (default)
    reranks the whole set. A set of <= 1 doc is a guaranteed no-op, so the
    reranker is not called at all.
    """

    def __init__(self, rerank_fn: RerankFn, max_candidates: int | None = None) -> None:
        self._rerank_fn = rerank_fn
        self._max_candidates = max_candidates

    def run(
        self, state: SearchAgentState, query: str | None = None
    ) -> list[SearchResult]:
        docs = list(state.retrieved_docs)
        if len(docs) <= 1:
            return docs
        if self._max_candidates is None or self._max_candidates >= len(docs):
            reordered = self._rerank_fn(query or state.question, docs)
        else:
            head = docs[: self._max_candidates]
            tail = docs[self._max_candidates :]
            reordered = self._rerank_fn(query or state.question, head) + tail
        state.record_rerank(reordered)
        return reordered
