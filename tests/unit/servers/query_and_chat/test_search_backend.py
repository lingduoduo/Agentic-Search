from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.auth import generate_user_jwt_token
from src.internal.db import AgenticSearchStore
from src.internal.search.process_search_query import SearchQueryResult
from src.internal.servers.query_and_chat.search_backend import create_search_router


def _client(tmp_path) -> tuple[TestClient, AgenticSearchStore]:
    store = AgenticSearchStore(tmp_path / "search.sqlite3")
    app = FastAPI()
    app.include_router(create_search_router(store))
    return TestClient(app), store


def _authenticated_headers() -> dict[str, str]:
    token = generate_user_jwt_token(
        user_id="alice",
        email="alice@example.com",
        group_ids=["engineering"],
    )
    return {"Authorization": f"Bearer {token}"}


def test_search_uses_server_acl_and_preserves_safe_caller_filters(
    tmp_path, monkeypatch
):
    captured = {}

    async def fake_run_expanded_search(query, **kwargs):
        captured["filters"] = kwargs["filters"]
        return SearchQueryResult(
            original_query=query,
            executed_queries=[query],
            results=[],
        )

    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.search_backend.run_expanded_search",
        fake_run_expanded_search,
    )
    client, store = _client(tmp_path)
    cutoff = "2026-01-02T03:04:05Z"

    response = client.post(
        "/search/send-search-message",
        headers=_authenticated_headers(),
        json={
            "search_query": "deployment guide",
            "stream": False,
            "filters": {
                "source_types": ["file"],
                "document_sets": ["engineering"],
                "tags": {"environment": "production"},
                "time_cutoff": cutoff,
            },
        },
    )

    assert response.status_code == 200
    filters = captured["filters"]
    assert filters.access_acl == [
        "email:alice@example.com",
        "group:engineering",
        "public",
        "user:alice",
    ]
    assert filters.source_types == ["file"]
    assert filters.document_sets == ["engineering"]
    assert filters.tags == {"environment": "production"}
    assert filters.time_cutoff == datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    store.close()


def test_search_discards_caller_supplied_acl(tmp_path, monkeypatch):
    captured = {}

    async def fake_run_expanded_search(query, **kwargs):
        captured["filters"] = kwargs["filters"]
        return SearchQueryResult(
            original_query=query,
            executed_queries=[query],
            results=[],
        )

    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.search_backend.run_expanded_search",
        fake_run_expanded_search,
    )
    client, store = _client(tmp_path)

    response = client.post(
        "/search/send-search-message",
        headers=_authenticated_headers(),
        json={
            "search_query": "private roadmap",
            "stream": False,
            "filters": {"access_acl": ["user:other"]},
        },
    )

    assert response.status_code == 200
    assert "user:other" not in captured["filters"].access_acl
    assert "user:alice" in captured["filters"].access_acl
    store.close()


def test_search_requires_authentication(tmp_path, monkeypatch):
    async def fake_run_expanded_search(*args, **kwargs):
        raise AssertionError("unauthenticated search must not execute")

    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.search_backend.run_expanded_search",
        fake_run_expanded_search,
    )
    client, store = _client(tmp_path)

    response = client.post(
        "/search/send-search-message",
        json={"search_query": "private roadmap", "stream": False},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required."}
    store.close()
