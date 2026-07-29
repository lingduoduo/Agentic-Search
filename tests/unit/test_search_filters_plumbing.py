"""End-to-end plumbing of per-user access `filters` into internal retrieval.

`SearchClient` already forwards `filters`; these tests lock that `search_tool`,
the `SearchAgentLoop`, and the web direct-search/agent entry points pass them
down for the internal corpus (and never for web providers, which have no ACL).
"""

from __future__ import annotations

import asyncio

import pytest

import src.internal.servers.web.app as web_app
import src.internal.tools.search as tools_search
from src.agents.search.search import (
    Retriever,
    SearchAgentLoop,
    SearchAgentLoopConfig,
)

_FILTERS = {"user_id": "user-A"}


# --- tools layer -----------------------------------------------------------


def test_search_tool_forwards_filters_for_retrieval(monkeypatch):
    captured: dict = {}

    async def _fake_retrieval_search(query, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(tools_search, "retrieval_search", _fake_retrieval_search)
    asyncio.run(
        tools_search.search_tool(
            "q", provider="retrieval", search_url="http://x", filters=_FILTERS
        )
    )
    assert captured["filters"] == _FILTERS


def test_search_tool_omits_filters_for_web_provider(monkeypatch):
    captured: dict = {}

    async def _fake_google(query, **kwargs):
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(tools_search, "google_custom_search", _fake_google)
    asyncio.run(tools_search.search_tool("q", provider="google", filters=_FILTERS))
    # Web providers never receive filters.
    assert "filters" not in captured["kwargs"]


# --- loop layer ------------------------------------------------------------


def _loop_with_fake_clients(filters):
    """Build a loop bypassing heavy __init__; only the retrieval bits are set."""
    loop = SearchAgentLoop.__new__(SearchAgentLoop)
    loop.search_config = SearchAgentLoopConfig(filters=filters)
    captured: dict = {}

    class _FakeClient:
        async def retrieve(self, queries, filters=None):
            captured["filters"] = filters
            return [[] for _ in queries]

    loop._search_client = _FakeClient()
    loop._web_search_client = _FakeClient()
    return loop, captured


def test_loop_threads_filters_to_vector_db_not_web():
    loop, captured = _loop_with_fake_clients(_FILTERS)

    asyncio.run(loop._retrieve_many(["q"], Retriever.VECTOR_DB))
    assert captured["filters"] == _FILTERS

    asyncio.run(loop._retrieve_many(["q"], Retriever.WEB))
    assert captured["filters"] is None


# --- web wiring ------------------------------------------------------------


def test_run_direct_search_forwards_filters(monkeypatch):
    captured: dict = {}

    async def _fake_search_tool(query, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(web_app, "search_tool", _fake_search_tool)
    asyncio.run(
        web_app._run_direct_search(
            "q",
            source_provider="retrieval",
            search_url="http://x",
            top_k=1,
            filters=_FILTERS,
        )
    )
    assert captured["filters"] == _FILTERS


def test_run_search_agent_threads_filters_into_config(monkeypatch):
    captured: dict = {}

    class _Stop(Exception):
        pass

    class _FakeLoop:
        def __init__(self, *, tokenizer, server_manager, search_config):
            captured["filters"] = search_config.filters
            raise _Stop()

    monkeypatch.setattr("src.get_registered_agent_loop", lambda name: _FakeLoop)

    with pytest.raises(_Stop):
        asyncio.run(
            web_app._run_search_agent(
                "q",
                manager=object(),
                tokenizer=object(),
                search_url="http://x",
                top_k=1,
                filters=_FILTERS,
            )
        )
    assert captured["filters"] == _FILTERS
