"""Access-control regression for the SEARCH auto-route.

`_run_search_direct_or_escalate` retrieves via `_run_direct_search` (which calls
`search_tool`) or escalates to `_run_search_agent` (the SearchAgentLoop). Neither
threads per-user access `filters` into retrieval, so when filters are present the
query must instead go through the filter-aware `_auto_search_pipeline`. These
tests lock that routing so a filtered request can never hit the unfiltered paths.
"""

from __future__ import annotations

import asyncio

import src.internal.servers.web.app as web_app


def _run(filters):
    called = {"direct": False, "agent": False, "pipeline_filters": "unset"}

    async def _fake_direct(*a, **k):
        called["direct"] = True
        return []

    async def _fake_agent(*a, **k):
        called["agent"] = True
        return ("agent answer", [], [], "search", {})

    async def _fake_pipeline(query, **kwargs):
        called["pipeline_filters"] = kwargs.get("filters")
        return ("filtered answer", [], [], "search", kwargs.get("extra", {}))

    return called, _fake_direct, _fake_agent, _fake_pipeline


def test_filters_present_routes_through_filtered_pipeline(monkeypatch):
    """With access filters, the unfiltered direct/agent paths are bypassed."""
    called, fake_direct, fake_agent, fake_pipeline = _run({"user_id": "userA"})
    monkeypatch.setattr(web_app, "_run_direct_search", fake_direct)
    monkeypatch.setattr(web_app, "_run_search_agent", fake_agent)
    monkeypatch.setattr(web_app, "_auto_search_pipeline", fake_pipeline)

    answer, _c, _d, intent, extra = asyncio.run(
        web_app._run_search_direct_or_escalate(
            "find the Q3 revenue deck",
            manager=object(),  # a "local model" is available…
            tokenizer=object(),
            llm=None,
            search_url="http://x/retrieve",
            browser_search_url=None,
            rerank_url=None,
            top_k=5,
            filters={"user_id": "userA"},
            history=[],
            source_provider="auto",
            on_turn=None,
        )
    )

    # …yet neither the direct-retrieval short-circuit nor the agent loop ran.
    assert called["direct"] is False
    assert called["agent"] is False
    # The filter-aware pipeline ran and received the filters.
    assert called["pipeline_filters"] == {"user_id": "userA"}
    assert answer == "filtered answer"
    assert intent == "search"
    assert extra.get("route_reason") == "access_filters_present"


def test_no_filters_preserves_direct_first_behavior(monkeypatch):
    """Without filters, the existing direct-first path is unchanged."""
    called, fake_direct, fake_agent, fake_pipeline = _run(None)
    monkeypatch.setattr(web_app, "_run_direct_search", fake_direct)
    monkeypatch.setattr(web_app, "_run_search_agent", fake_agent)
    monkeypatch.setattr(web_app, "_auto_search_pipeline", fake_pipeline)

    asyncio.run(
        web_app._run_search_direct_or_escalate(
            "FAISS",
            manager=object(),
            tokenizer=object(),
            llm=None,
            search_url="http://x/retrieve",
            browser_search_url=None,
            rerank_url=None,
            top_k=5,
            filters=None,
            history=[],
            source_provider="retrieval",
            on_turn=None,
        )
    )

    # The filtered-pipeline shortcut did not fire; direct retrieval ran.
    assert called["pipeline_filters"] == "unset"
    assert called["direct"] is True
