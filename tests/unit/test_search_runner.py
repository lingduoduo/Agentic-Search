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


async def test_build_search_contexts_pads_a_short_response_and_logs(caplog):
    """A response with fewer rows than queries must not silently lose queries.

    Reproduces the shape SearchClient.retrieve's single-query response
    collapse (client.py's `if rows and isinstance(rows[0], dict): rows =
    [rows]`) can produce for a batched request: 2 rows for 3 queries. The
    query without a row must still get a bundle (empty, not missing) and the
    gap must be logged, not silently swallowed.
    """
    import logging
    from unittest.mock import patch

    from src.context.retrieval.search_runner import build_search_contexts
    from src.context.search import SearchResult

    async def _fake_retrieve(self, queries, topk=None, filters=None):
        # Server returns only 2 rows for the 3 requested queries.
        return [
            [SearchResult(contents="about alpha", title="alpha", url="http://x/a")],
            [SearchResult(contents="about beta", title="beta", url="http://x/b")],
        ]

    with (
        caplog.at_level(logging.WARNING),
        patch("src.context.retrieval.client.SearchClient.retrieve", _fake_retrieve),
    ):
        bundles = await build_search_contexts(
            ["alpha", "beta", "gamma"], top_k=5, search_url="http://x/retrieve"
        )

    assert [b.query for b in bundles] == ["alpha", "beta", "gamma"]
    assert bundles[0].documents and bundles[0].documents[0].content.endswith("alpha")
    assert bundles[1].documents and bundles[1].documents[0].content.endswith("beta")
    assert bundles[2].documents == [], (
        "gamma's missing row must yield an empty bundle, not a dropped query"
    )
    assert any("2 rows for 3 queries" in record.message for record in caplog.records), (
        "the row/query mismatch must be logged, not silent"
    )


async def test_build_search_contexts_enforces_filters_locally_on_every_query():
    """The batched path must re-apply filters to returned rows, not just forward them.

    Regression for PRs #487-#492's hard invariant: a third-party backend need
    not honour a forwarded filter, and anything it returns reaches a model's
    context, so filtering after the fact is the only safe enforcement point.
    Guards against a future refactor that inlines build_context_bundle over
    `rows` directly and drops the `_apply_filters` call -- every existing
    batched-path test passes `filters=None`, so none of them would catch that.
    """
    from unittest.mock import patch

    from src.context.models import SearchFilters
    from src.context.retrieval.search_runner import build_search_contexts
    from src.context.search import SearchResult

    async def _fake_retrieve(self, queries, topk=None, filters=None):
        return [
            [
                SearchResult(
                    title=f"Visible {q}",
                    contents=f'"Visible {q}"\nAllowed',
                    metadata={"acl": ["public"]},
                ),
                SearchResult(
                    title=f"Blocked {q}",
                    contents=f'"Blocked {q}"\nHidden',
                    metadata={"acl": ["user:alice"]},
                ),
            ]
            for q in queries
        ]

    with patch("src.context.retrieval.client.SearchClient.retrieve", _fake_retrieve):
        bundles = await build_search_contexts(
            ["alpha", "beta"],
            top_k=5,
            filters=SearchFilters(access_acl=["public"]),
            search_url="http://x/retrieve",
        )

    assert [b.query for b in bundles] == ["alpha", "beta"]
    for bundle in bundles:
        titles = [doc.title for doc in bundle.documents]
        assert titles, "the allowed document must still be present"
        assert all(not title.startswith("Blocked") for title in titles), (
            f"a document outside the filter reached the bundle: {titles}"
        )
