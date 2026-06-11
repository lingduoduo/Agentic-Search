"""Search query processing: LLM expansion, flow classification, parallel search, RRF merge."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from src.context.models import ChatMessage
from src.context.models import LLMClient
from src.context.models import LLMResponse
from src.context.models import SearchFilters
from src.context.models import SearchRequest
from src.context.retrieval.search_runner import run_search
from src.internal.prompts.query_expansion import QUERY_TYPE_PROMPT
from src.internal.prompts.search_flow_classification import CHAT_CLASS
from src.internal.prompts.search_flow_classification import SEARCH_CLASS
from src.context.search import SearchResult
from src.internal.servers.secondary_llm_flows import classify_is_search_flow
from src.internal.servers.secondary_llm_flows import expand_keywords

logger = logging.getLogger(__name__)

_RRF_K = 60
_ORIGINAL_QUERY_WEIGHT = 2.0
_EXPANSION_WEIGHT = 1.0


@dataclass(frozen=True)
class SearchQueryResult:
    """Outcome of a (possibly expanded) search."""

    original_query: str
    executed_queries: list[str]
    results: list[SearchResult]
    query_type: str | None = None
    flow_class: str | None = None


def _llm_text(response: LLMResponse | str) -> str:
    return response if isinstance(response, str) else response.content


def classify_query_type(user_query: str, llm: LLMClient) -> str:
    """Return ``'keyword'`` or ``'semantic'`` for the query."""
    prompt = QUERY_TYPE_PROMPT.format(user_query=user_query)
    text = _llm_text(llm.complete([ChatMessage(role="user", content=prompt)])).lower()
    return "keyword" if "keyword" in text else "semantic"


def classify_search_flow(user_query: str, llm: LLMClient) -> str:
    """Return ``'search'`` or ``'chat'`` for the query."""
    return SEARCH_CLASS if classify_is_search_flow(user_query, llm) else CHAT_CLASS


def weighted_reciprocal_rank_fusion(
    ranked_results: list[list[SearchResult]],
    weights: list[float],
    *,
    k: int = _RRF_K,
) -> list[SearchResult]:
    """Merge ranked result lists using weighted Reciprocal Rank Fusion.

    Each result is keyed by (url, first-100-chars-of-contents). The best
    original score is kept for the winning entry. Results are returned in
    descending RRF score order.
    """
    rrf_scores: dict[tuple[str | None, str], float] = {}
    best: dict[tuple[str | None, str], SearchResult] = {}

    for result_list, weight in zip(ranked_results, weights):
        for rank, result in enumerate(result_list):
            key = (result.url, result.contents[:100])
            rrf_scores[key] = rrf_scores.get(key, 0.0) + weight / (k + rank + 1)
            if key not in best or result.score > best[key].score:
                best[key] = result

    return sorted(
        best.values(), key=lambda r: rrf_scores[(r.url, r.contents[:100])], reverse=True
    )


async def run_expanded_search(
    query: str,
    *,
    llm: LLMClient | None = None,
    search_url: str = "http://localhost:8000/retrieve",
    top_k: int = 5,
    filters: SearchFilters | None = None,
    expand: bool = True,
) -> SearchQueryResult:
    """Run query expansion then parallel search, merging results with weighted RRF.

    When ``expand=False`` or no LLM is provided, a single search is executed.
    Expansion failures are logged and silently skipped — the original query
    always runs. Each parallel search failure is also skipped rather than
    aborting the entire request.
    """
    keyword_expansions: list[str] = []
    if llm is not None and expand:
        keyword_expansions = expand_keywords(query, llm)
        if keyword_expansions:
            logger.debug(
                "Query expansion produced %d keyword queries for: %r",
                len(keyword_expansions),
                query,
            )

    all_queries = [query] + keyword_expansions

    tasks = [
        run_search(
            SearchRequest(query=q, top_k=top_k, filters=filters),
            search_url=search_url,
        )
        for q in all_queries
    ]
    raw: list[list[SearchResult] | BaseException] = await asyncio.gather(
        *tasks, return_exceptions=True
    )

    results_per_query: list[list[SearchResult]] = []
    for q, outcome in zip(all_queries, raw):
        if isinstance(outcome, BaseException):
            logger.warning("Search failed for query %r: %s", q, outcome)
        else:
            results_per_query.append(outcome)

    if not results_per_query:
        merged: list[SearchResult] = []
    elif len(results_per_query) == 1:
        merged = results_per_query[0]
    else:
        weights = [_ORIGINAL_QUERY_WEIGHT] + [_EXPANSION_WEIGHT] * (
            len(results_per_query) - 1
        )
        merged = weighted_reciprocal_rank_fusion(results_per_query, weights)

    return SearchQueryResult(
        original_query=query,
        executed_queries=all_queries,
        results=merged[:top_k],
    )


__all__ = [
    "SearchQueryResult",
    "classify_query_type",
    "classify_search_flow",
    "expand_keywords",
    "run_expanded_search",
    "weighted_reciprocal_rank_fusion",
]
