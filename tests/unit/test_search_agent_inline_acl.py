"""The search agent filters before the model reads, not after.

`SearchAgentLoop` feeds retrieved documents into the model's context across
turns. Filtering the documents it *returns* is too late — the answer has already
been written from whatever the retriever handed over. These tests assert on what
the loop keeps at the retrieval boundary, so they fail if the post-filter is
dropped even though `app.py`'s later `_enforce_access` would still clean the
returned list.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from src.agents.search import SearchAgentLoop, SearchAgentLoopConfig
from src.context.models import SearchFilters
from src.context.search import SearchResult


def _result(doc_id: str, acl: list[str] | None) -> SearchResult:
    return SearchResult(
        contents=f"{doc_id} body",
        url=f"http://x/{doc_id}",
        title=doc_id,
        metadata={"acl": acl} if acl is not None else {"source": "corpus"},
    )


ROW = [
    _result("mine", ["user:userA"]),
    _result("theirs", ["user:userB"]),
    _result("open", ["public"]),
    _result("undeclared", None),
]


def _loop(filters):
    loop = SearchAgentLoop(
        tokenizer=MagicMock(),
        server_manager=MagicMock(),
        search_config=SearchAgentLoopConfig(filters=filters),
    )
    client = MagicMock()
    client.retrieve = AsyncMock(return_value=[list(ROW)])
    loop._search_client = client
    return loop, client


def _titles(rows):
    return {r.title for r in rows[0]}


def test_documents_outside_the_acl_never_enter_the_loop():
    loop, _client = _loop(SearchFilters(access_acl=["public", "user:userA"]))
    rows = asyncio.run(loop._retrieve_many(["q"]))
    titles = _titles(rows)
    assert "theirs" not in titles
    assert {"mine", "open"} <= titles


def test_documents_with_no_declared_acl_stay_public():
    loop, _client = _loop(SearchFilters(access_acl=["public"]))
    assert "undeclared" in _titles(asyncio.run(loop._retrieve_many(["q"])))


def test_no_filters_keeps_every_document():
    loop, _client = _loop(None)
    assert len(asyncio.run(loop._retrieve_many(["q"]))[0]) == len(ROW)


def test_the_payload_sent_to_the_client_is_json_serializable():
    # Passing the object here is what crashed every internal retrieval call in
    # #487 and masked its enforcement as "internal unreachable".
    loop, client = _loop(SearchFilters(access_acl=["public", "user:userA"]))
    asyncio.run(loop._retrieve_many(["q"]))
    sent = client.retrieve.await_args.kwargs["filters"]
    assert json.dumps(sent)  # raises if it is not serializable
    assert sent == {"access_acl": ["public", "user:userA"]}


def test_no_filters_sends_no_payload():
    loop, client = _loop(None)
    asyncio.run(loop._retrieve_many(["q"]))
    assert client.retrieve.await_args.kwargs["filters"] is None


def test_web_retrieval_is_never_filtered():
    # Web results carry no ACL; filtering them would drop everything.
    from src.agents.search.search import Retriever

    loop, _client = _loop(SearchFilters(access_acl=["public"]))
    web = MagicMock()
    web.retrieve = AsyncMock(return_value=[list(ROW)])
    loop._web_search_client = web

    rows = asyncio.run(loop._retrieve_many(["q"], retriever=Retriever.WEB))
    assert len(rows[0]) == len(ROW)
    assert web.retrieve.await_args.kwargs["filters"] is None


def test_a_retrieval_failure_still_degrades_to_empty():
    loop, client = _loop(SearchFilters(access_acl=["public"]))
    client.retrieve = AsyncMock(side_effect=ConnectionError("no route"))
    assert asyncio.run(loop._retrieve_many(["q", "q2"])) == [[], []]
