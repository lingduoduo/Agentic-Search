"""The corpus search tool returns only what the caller may read.

demo.py and hybrid.py accept the `filters` field and ignore it, so sending the
payload is not enforcement — the tool drops inaccessible pages itself.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

from src.context.models import SearchFilters
from src.internal.tools.routing_tools import build_search_routing_tool
from src.internal.tools.search import SearchPage


def _page(title: str, acl: list[str] | None) -> SearchPage:
    return SearchPage(
        title=title,
        summary="body",
        url=f"http://x/{title}",
        metadata={"acl": acl if acl is not None else ["public"]},
    )


PAGES = [
    _page("mine", ["user:userA"]),
    _page("theirs", ["user:userB"]),
    _page("open", ["public"]),
    _page("undeclared", None),
]


def _run(filters):
    tool = build_search_routing_tool(
        search_url="http://x/retrieve", top_k=5, filters=filters
    )
    with patch(
        "src.internal.tools.routing_tools.search_tool",
        new=AsyncMock(return_value=list(PAGES)),
    ) as mock:
        raw, _out, _meta = asyncio.run(tool.execute("id", {"query": "q"}))
    return json.loads(raw), mock


def test_documents_outside_the_acl_are_dropped():
    results, _ = _run(SearchFilters(access_acl=["public", "user:userA"]))
    titles = {r["title"] for r in results}
    assert "theirs" not in titles
    assert {"mine", "open"} <= titles


def test_documents_with_no_declared_acl_stay_public():
    results, _ = _run(SearchFilters(access_acl=["public"]))
    assert "undeclared" in {r["title"] for r in results}


def test_no_filters_returns_everything():
    # Training scripts and evals build this tool without an identity.
    results, _ = _run(None)
    assert len(results) == len(PAGES)


def test_the_payload_is_sent_to_retrieval():
    # Backends that honour access_acl should get the chance to.
    _results, mock = _run(SearchFilters(access_acl=["public", "user:userA"]))
    sent = mock.await_args.kwargs["filters"]
    assert sent == {"access_acl": ["public", "user:userA"]}


def test_no_filters_sends_no_payload():
    _results, mock = _run(None)
    assert mock.await_args.kwargs["filters"] is None


def test_everything_filtered_out_is_an_empty_list_not_an_error():
    # An empty result is a legitimate answer; it must not read as a failure.
    results, _ = _run(SearchFilters(access_acl=["user:nobody"]))
    assert results == []
