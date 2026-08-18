"""Search: query orchestration helpers and the session-aware search pipeline.

This package is the merger of the former ``search`` (query orchestration --
classification, keyword expansion, expanded search, RRF) and ``search_pipeline``
(the staged retrieve -> rank -> answer pipeline and its models). Both halves
describe one search path, so they live under one name.

Every module here is light -- dataclasses, ``httpx`` and ``src.context``. Nothing
below pulls in a heavy ML dependency, which is why these re-exports can stay
eager; consumers that import a submodule directly (``.models``, ``.stages``)
still run this file first, so anything expensive added here would be paid for by
all of them.
"""

from .context import RetrievalContext, build_retrieval_context
from .pipeline import SearchPipeline
from .process_search_query import (
    SearchQueryResult,
    classify_query_type,
    classify_search_flow,
    expand_keywords,
    run_expanded_search,
    weighted_reciprocal_rank_fusion,
)

__all__ = [
    "RetrievalContext",
    "SearchPipeline",
    "SearchQueryResult",
    "build_retrieval_context",
    "classify_query_type",
    "classify_search_flow",
    "expand_keywords",
    "run_expanded_search",
    "weighted_reciprocal_rank_fusion",
]
