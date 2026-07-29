"""Search runners that build normalized context bundles."""

from __future__ import annotations

from src.context.retrieval.client import SearchClient
from src.context.retrieval.client import SearchClientConfig
from src.context.search import SearchResult
from src.internal.tools.search import google_custom_search
from src.internal.tools.search import serpapi_search

from ..enums import SearchType
from ..models import SearchRequest
from ..utils import build_context_bundle


async def run_search(
    request: SearchRequest,
    *,
    search_url: str = "http://localhost:8000/retrieve",
    timeout_seconds: int = 15,
    max_retries: int = 3,
    fetch_url: str | None = None,
) -> list[SearchResult]:
    request.validate()
    if request.provider == SearchType.RETRIEVAL:
        client = SearchClient(
            SearchClientConfig(
                url=search_url,
                topk=request.top_k,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                fetch_url=fetch_url,
            )
        )
        try:
            if request.filters:
                results = await client.retrieve_one(
                    request.query,
                    topk=request.top_k,
                    filters=request.filters.to_payload(),
                )
                return _apply_filters(results, request.filters)
            return await client.retrieve_one(request.query, topk=request.top_k)
        finally:
            await client.aclose()

    if request.provider == SearchType.GOOGLE:
        pages = await google_custom_search(
            request.query,
            page_size=request.top_k,
            timeout_seconds=timeout_seconds,
        )
    elif request.provider == SearchType.SERPAPI:
        pages = await serpapi_search(
            request.query,
            page_size=request.top_k,
            timeout_seconds=timeout_seconds,
        )
    else:
        raise ValueError(f"Unsupported search provider: {request.provider}")

    results = [
        SearchResult(
            title=page.title,
            contents=f'"{page.title}"\n{page.summary}' if page.title else page.summary,
            url=page.url,
        )
        for page in pages
        if not page.error
    ]
    return _apply_filters(results, request.filters)


async def build_search_context(
    request: SearchRequest,
    *,
    search_url: str = "http://localhost:8000/retrieve",
    timeout_seconds: int = 15,
    max_retries: int = 3,
    fetch_url: str | None = None,
):
    results = await run_search(
        request,
        search_url=search_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        fetch_url=fetch_url,
    )
    return build_context_bundle(
        request.query,
        _apply_filters(results, request.filters),
        max_documents=request.top_k,
    )


def combine_search_results(result_sets: list[list[SearchResult]]) -> list[SearchResult]:
    unique: dict[tuple[str | None, str], SearchResult] = {}
    for result_set in result_sets:
        for result in result_set:
            key = (result.url, result.contents)
            previous = unique.get(key)
            if previous is None or result.score > previous.score:
                unique[key] = result
    return sorted(unique.values(), key=lambda result: result.score, reverse=True)


def _apply_filters(
    results: list[SearchResult],
    filters,
) -> list[SearchResult]:
    if filters is None:
        return results
    return [result for result in results if filters.matches(result.metadata)]
