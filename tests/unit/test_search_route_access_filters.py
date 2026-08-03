"""Access filters narrow results; they do not change which route runs.

Signing in used to divert every query away from the direct-first path into
`_auto_search_pipeline`, because `_run_direct_search` and `_run_search_agent`
once took no `filters` and would have retrieved across every user's documents.
Both thread filters end-to-end now (#407), so that diversion only made an
authenticated query slower and differently-answered than the same query
anonymously.

These tests lock the intended contract: the same query takes the same route
whether or not a user is signed in, and every path that retrieves receives the
filters.
"""

from __future__ import annotations

import asyncio

import src.internal.servers.web.app as web_app

FILTERS = {"access_acl": ["public", "user:userA"]}


def _harness(*, direct_documents=None):
    """Fakes for the three retrieval paths, recording the filters each receives."""
    seen: dict = {
        "direct_filters": "uncalled",
        "agent_filters": "uncalled",
        "pipeline_filters": "uncalled",
    }

    async def _fake_direct(query, **kwargs):
        seen["direct_filters"] = kwargs.get("filters")
        return list(direct_documents or [])

    async def _fake_agent(query, **kwargs):
        seen["agent_filters"] = kwargs.get("filters")
        return ("agent answer", [], [], "search", {})

    async def _fake_pipeline(query, **kwargs):
        seen["pipeline_filters"] = kwargs.get("filters")
        return ("pipeline answer", [], [], "search", kwargs.get("extra", {}))

    return seen, _fake_direct, _fake_agent, _fake_pipeline


def _call(monkeypatch, *, filters, manager=object(), tokenizer=object(), **overrides):
    seen, fake_direct, fake_agent, fake_pipeline = _harness(**overrides)
    monkeypatch.setattr(web_app, "_run_direct_search", fake_direct)
    monkeypatch.setattr(web_app, "_run_search_agent", fake_agent)
    monkeypatch.setattr(web_app, "_auto_search_pipeline", fake_pipeline)

    result = asyncio.run(
        web_app._run_search_direct_or_escalate(
            "FAISS",
            manager=manager,
            tokenizer=tokenizer,
            llm=None,
            search_url="http://x/retrieve",
            browser_search_url=None,
            rerank_url=None,
            top_k=5,
            filters=filters,
            history=[],
            source_provider="retrieval",
            on_turn=None,
        )
    )
    return seen, result


def test_signed_in_query_still_starts_with_direct_retrieval(monkeypatch):
    seen, (_a, _c, _d, _i, extra) = _call(monkeypatch, filters=FILTERS)

    # The direct-first path runs for an authenticated user, exactly as it does
    # anonymously — signing in must not swap the pipeline.
    assert seen["direct_filters"] == FILTERS
    assert seen["pipeline_filters"] == "uncalled"
    assert extra.get("route_reason") != "access_filters_present"


def test_anonymous_and_signed_in_take_the_same_route(monkeypatch):
    anon_seen, (_a1, _c1, _d1, _i1, anon_extra) = _call(monkeypatch, filters=None)
    auth_seen, (_a2, _c2, _d2, _i2, auth_extra) = _call(monkeypatch, filters=FILTERS)

    assert anon_extra.get("search_mode") == auth_extra.get("search_mode")
    assert anon_seen["direct_filters"] is None
    assert auth_seen["direct_filters"] == FILTERS  # narrowed, not rerouted


def test_escalation_receives_the_filters(monkeypatch):
    # Direct retrieval finds nothing, so the gate is weak and the agent runs.
    seen, _ = _call(monkeypatch, filters=FILTERS)
    assert seen["agent_filters"] == FILTERS


def test_degraded_pipeline_receives_the_filters(monkeypatch):
    # No local model: escalation falls back to _auto_search_pipeline, which must
    # still be filtered.
    seen, _ = _call(monkeypatch, filters=FILTERS, manager=None, tokenizer=None)
    assert seen["pipeline_filters"] == FILTERS


def test_no_filters_preserves_direct_first_behavior(monkeypatch):
    seen, _ = _call(monkeypatch, filters=None)
    assert seen["direct_filters"] is None
    assert seen["pipeline_filters"] == "uncalled"


# ---------------------------------------------------------------------------
# Enforcement, not just plumbing. demo.py and hybrid.py ignore the `filters`
# they are sent — only the full RetrievalService honours them — so the route
# must drop inaccessible documents itself rather than trusting the server.
# ---------------------------------------------------------------------------


def _doc(doc_id: str, acl: list[str] | None):
    from src.context.models import ContextDocument

    # Title matches the query so the direct gate fires and the path returns
    # documents rather than escalating to the (faked, empty) agent.
    return ContextDocument(
        id=doc_id,
        title="FAISS",
        content="c",
        url=None,
        score=0.9,
        metadata={"acl": acl} if acl is not None else {},
    )


def test_documents_outside_the_acl_never_reach_the_caller(monkeypatch):
    from src.context.models import SearchFilters

    mine, theirs, public = (
        _doc("mine", ["user:userA"]),
        _doc("theirs", ["user:userB"]),
        _doc("public", ["public"]),
    )
    seen, (_a, _c, documents, _i, _e) = _call(
        monkeypatch,
        filters=SearchFilters(access_acl=["public", "user:userA"]),
        direct_documents=[mine, theirs, public],
    )

    returned = {d.id for d in documents}
    assert "theirs" not in returned  # another user's document
    assert {"mine", "public"} <= returned


def test_unfiltered_requests_keep_every_document(monkeypatch):
    docs = [_doc("a", ["user:userB"]), _doc("b", None)]
    _seen, (_a, _c, documents, _i, _e) = _call(
        monkeypatch, filters=None, direct_documents=docs
    )
    assert {d.id for d in documents} == {"a", "b"}


def test_escalated_documents_are_enforced_too(monkeypatch):
    # The SearchAgentLoop retrieves through the same client, so a server that
    # ignores filters would leak here as well.
    from src.context.models import SearchFilters

    async def _fake_direct(query, **kwargs):
        return []  # weak gate -> escalate

    async def _fake_agent(query, **kwargs):
        return (
            "agent answer",
            [],
            [_doc("mine", ["user:userA"]), _doc("theirs", ["user:userB"])],
            "search",
            {},
        )

    monkeypatch.setattr(web_app, "_run_direct_search", _fake_direct)
    monkeypatch.setattr(web_app, "_run_search_agent", _fake_agent)

    _a, _c, documents, _i, _e = asyncio.run(
        web_app._run_search_direct_or_escalate(
            "FAISS",
            manager=object(),
            tokenizer=object(),
            llm=None,
            search_url="http://x/retrieve",
            browser_search_url=None,
            rerank_url=None,
            top_k=5,
            filters=SearchFilters(access_acl=["public", "user:userA"]),
            history=[],
            source_provider="retrieval",
            on_turn=None,
        )
    )
    assert {d.id for d in documents} == {"mine"}


def test_anonymous_callers_are_public_only():
    # Anonymous used to carry no ACL at all, so a document restricted to
    # another user was readable by anyone logged out.
    from src.internal.access.capabilities import resolve_capabilities

    class _Store:
        def get_user_memories(self, user_id):
            return []

    caps = resolve_capabilities(None, _Store())
    assert caps.access_acl == ["public"]


def test_a_restricted_document_is_hidden_from_anonymous(monkeypatch):
    from src.context.models import SearchFilters

    restricted = _doc("theirs", ["user:someone_else"])
    public = _doc("public", ["public"])
    _seen, (_a, _c, documents, _i, _e) = _call(
        monkeypatch,
        filters=SearchFilters(access_acl=["public"]),
        direct_documents=[restricted, public],
    )
    assert {d.id for d in documents} == {"public"}
