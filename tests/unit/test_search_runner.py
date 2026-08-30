async def test_build_search_contexts_issues_one_request_for_all_queries():
    """N queries cost one round trip, not N.

    Guards the batch: the retrieval API takes {"queries": [...]}, and issuing
    one single-query request per query paid N session setups.
    """
    from unittest.mock import patch

    from src.context.retrieval.search_runner import build_search_contexts
    from src.context.search import SearchResult

    calls: list[list[str]] = []

    async def _fake_retrieve(self, queries, topk=None, filters=None):
        calls.append(list(queries))
        return [
            [SearchResult(contents=f"about {q}", title=q, url=f"http://x/{q}")]
            for q in queries
        ]

    with patch("src.context.retrieval.client.SearchClient.retrieve", _fake_retrieve):
        bundles = await build_search_contexts(
            ["alpha", "beta", "gamma"], top_k=5, search_url="http://x/retrieve"
        )

    assert len(calls) == 1, f"expected one batched request, got {len(calls)}"
    assert calls[0] == ["alpha", "beta", "gamma"]
    assert [b.query for b in bundles] == ["alpha", "beta", "gamma"]
    assert bundles[0].documents[0].content.endswith("alpha")


async def test_build_search_contexts_returns_empty_bundles_for_no_queries():
    from src.context.retrieval.search_runner import build_search_contexts

    assert await build_search_contexts([], search_url="http://x/retrieve") == []
