"""The serialization boundary, crossed for real.

`filters` has two shapes: the `SearchFilters` object `_enforce_access` needs for
its `.matches()` check, and the plain dict every retrieval call puts on the
wire. The existing access-filter tests monkeypatch the runners with fakes that
accept either, so a call site that shipped the object where JSON was expected
passed them and still failed in production.

These tests patch one level lower — at `search_tool` and at the runner
boundary — and assert two things that must always hold together:

1. what each call site passes is `json.dumps`-able, and
2. the documents that come back are enforced.

Serializing without enforcing is the more dangerous half: the retrieval servers
this project ships (`demo.py`, `hybrid.py`) accept a `filters` field and ignore
it, so an unpaired payload turns a fail-closed crash into a silent leak.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.internal.servers.web.app import SearchExperienceSettings, create_web_app
from src.internal.tools.search import SearchPage

RESTRICTED = SearchPage(
    title="Zebra Handbook",
    summary="Zebra confidential notes.",
    url="https://x.test/secret",
    score=0.95,
    metadata={"acl": ["user:someone_else"]},
)
PUBLIC = SearchPage(
    title="Zebra Handbook",
    summary="Zebra migration patterns.",
    url="https://x.test/public",
    score=0.9,
    metadata={"acl": ["public"]},
)


def _client(tmp_path, name: str) -> TestClient:
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / f"{name}.sqlite3"))
    return TestClient(app)


def _patch_retrieval(monkeypatch, seen: list, pages: list[SearchPage]):
    """Stand in for the retrieval server, recording the filters as JSON text.

    `json.dumps` here is the assertion: it raises on a `SearchFilters` object,
    which is exactly what `retrieval_search` does before retrying three times
    and returning an error card.
    """

    async def _fake_search_tool(query, *, provider, search_url, page_size, **kwargs):
        if provider == "retrieval":
            seen.append(json.dumps(kwargs.get("filters")))
            return list(pages)
        return []

    async def _fake_fetch_pages(pgs, **kwargs):
        return pgs

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", _fake_fetch_pages
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.expand_keywords", lambda query, llm: []
    )
    return seen


def _documents(response) -> list[dict]:
    assert response.status_code == 200, response.text
    return response.json().get("documents") or []


# --- mode=search_tool -------------------------------------------------------


def test_search_tool_mode_sends_serializable_filters_and_enforces(
    monkeypatch, tmp_path
):
    seen: list[str] = []
    _patch_retrieval(monkeypatch, seen, [RESTRICTED, PUBLIC])

    documents = _documents(
        _client(tmp_path, "search_tool").post(
            "/api/agent",
            json={
                "query": "Zebra Handbook",
                "mode": "search_tool",
                "source_provider": "retrieval",
            },
        )
    )

    # Anonymous is ["public"], never absent: an empty ACL would read as
    # "unfiltered" to the retrieval server.
    assert seen == [json.dumps({"access_acl": ["public"]})]
    contents = " ".join(d["content"] for d in documents).lower()
    assert "migration" in contents  # the public document did come back
    assert "confidential" not in contents


# --- mode=hybrid_search -----------------------------------------------------


def test_hybrid_search_mode_sends_serializable_filters_and_enforces(
    monkeypatch, tmp_path
):
    seen: list[str] = []
    _patch_retrieval(monkeypatch, seen, [RESTRICTED, PUBLIC])

    # `source_provider="all"` takes the multi-leg fan-out inside
    # `_run_hybrid_search`, not the single-provider path that
    # `run_expanded_search` already filters. That fan-out was the live bypass:
    # it sent `_filters_payload(filters)` and checked nothing on the way back.
    documents = _documents(
        _client(tmp_path, "hybrid").post(
            "/api/agent",
            json={
                "query": "Zebra Handbook",
                "mode": "hybrid_search",
                "source_provider": "all",
            },
        )
    )

    assert seen and all(s == json.dumps({"access_acl": ["public"]}) for s in seen)
    contents = " ".join(d["content"] for d in documents).lower()
    assert "migration" in contents
    assert "confidential" not in contents


def test_hybrid_search_enforces_on_the_auto_fan_out_too(monkeypatch, tmp_path):
    # "auto" runs the same retrieval leg beside the web cascade; it is reachable
    # with a plain {"query": ..., "mode": "hybrid_search"} body.
    seen: list[str] = []
    _patch_retrieval(monkeypatch, seen, [RESTRICTED, PUBLIC])

    documents = _documents(
        _client(tmp_path, "hybrid_auto").post(
            "/api/agent",
            json={"query": "Zebra Handbook", "mode": "hybrid_search"},
        )
    )

    contents = " ".join(d["content"] for d in documents).lower()
    assert "confidential" not in contents


# --- mode=search_agent ------------------------------------------------------


def test_search_agent_mode_sends_serializable_filters_and_enforces(
    monkeypatch, tmp_path
):
    """The agent loop puts `filters` straight into `SearchAgentLoopConfig`, which
    posts it as JSON, so the route must hand it a dict — and check what returns."""
    from src.context.models import ContextDocument

    seen: list[str] = []

    def _doc(doc_id: str, acl: list[str]) -> ContextDocument:
        return ContextDocument(
            id=doc_id,
            title="Zebra Handbook",
            content="Zebra confidential notes." if "else" in acl[0] else "Zebra notes.",
            url=None,
            score=0.9,
            metadata={"acl": acl},
        )

    async def _fake_agent(query, **kwargs):
        seen.append(json.dumps(kwargs.get("filters")))
        docs = [_doc("theirs", ["user:someone_else"]), _doc("mine", ["public"])]
        return ("agent answer", [d.citation for d in docs], docs, "search", {})

    monkeypatch.setattr("src.internal.servers.web.app._run_search_agent", _fake_agent)

    client = _client(tmp_path, "search_agent")
    client.app.state.search_agent_manager = object()
    client.app.state.search_agent_tokenizer = object()

    response = client.post(
        "/api/agent", json={"query": "Zebra Handbook", "mode": "search_agent"}
    )
    documents = _documents(response)

    assert seen == [json.dumps({"access_acl": ["public"]})]
    contents = " ".join(d["content"] for d in documents).lower()
    assert "confidential" not in contents
    # Citations name only what survived; the loop's own list still named the
    # document that was just dropped.
    citations = response.json()["messages"][-1]["metadata"].get("citations") or []
    assert all("theirs" not in str(c) for c in citations)
