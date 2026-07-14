"""Tests for src/servers/query_and_chat and src/servers/documents."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.internal.auth import generate_user_jwt_token
from src.internal.configs import AppSettings
from src.internal.configs import AuthSettings
from src.internal.db import AgenticSearchStore
from src.internal.db.models import ConnectorConfig
from src.internal.db.models import UserRecord
from src.context.search import SearchResult
from src.internal.servers.query_and_chat.models import SearchDocWithContent
from src.internal.servers.query_and_chat.models import SendSearchQueryRequest
from src.internal.servers.query_and_chat.streaming_models import SearchErrorPacket
from src.internal.servers.query_and_chat.streaming_models import SearchQueriesPacket
from src.internal.servers.query_and_chat.token_limit import check_message_rate_limit
from src.internal.servers.query_and_chat.token_limit import reset_rate_limit
from src.internal.servers.web.app import SearchExperienceSettings
from src.internal.servers.web.app import create_web_app

_ADMIN_ID = "admin-user"
_USER_ID = "regular-user"


def _bearer(user_id: str) -> dict:
    return {"Authorization": f"Bearer {generate_user_jwt_token(user_id=user_id)}"}


def _admin_client(tmp_path):
    store = AgenticSearchStore(tmp_path / "test.sqlite3")
    store.upsert_user(UserRecord(id=_ADMIN_ID, email="admin@test.local"))
    store.upsert_user(UserRecord(id=_USER_ID, email="user@test.local"))
    settings = AppSettings(auth=AuthSettings(super_users=(_ADMIN_ID,)))
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "test.sqlite3"),
        app_settings=settings,
        store=store,
    )
    return TestClient(app), store


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def test_search_doc_with_content_from_search_result():
    result = SearchResult(
        contents="some content",
        score=0.87,
        title="Doc Title",
        url="https://example.com/doc",
        metadata={"source_type": "file"},
    )
    doc = SearchDocWithContent.from_search_result(result)
    assert doc.content == "some content"
    assert doc.score == pytest.approx(0.87)
    assert doc.title == "Doc Title"
    assert doc.url == "https://example.com/doc"
    assert doc.metadata["source_type"] == "file"


def test_send_search_query_request_defaults():
    req = SendSearchQueryRequest(search_query="what is ML?")
    assert req.num_hits == 10
    assert req.run_query_expansion is False
    assert req.stream is False
    assert req.filters is None


# ---------------------------------------------------------------------------
# streaming_models
# ---------------------------------------------------------------------------


def test_streaming_packets_serialize_with_type_discriminator():
    q_packet = SearchQueriesPacket(all_executed_queries=["q1", "q2"])
    assert q_packet.type == "search_queries"
    assert "all_executed_queries" in q_packet.model_dump()

    e_packet = SearchErrorPacket(error="oops")
    assert e_packet.type == "search_error"


# ---------------------------------------------------------------------------
# token_limit
# ---------------------------------------------------------------------------


def test_rate_limit_allows_up_to_max(tmp_path):
    uid = "rate-test-user"
    reset_rate_limit(uid)
    for _ in range(3):
        check_message_rate_limit(uid, max_messages=3, window_seconds=60)
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        check_message_rate_limit(uid, max_messages=3, window_seconds=60)
    assert exc_info.value.status_code == 429
    reset_rate_limit(uid)


def test_rate_limit_disabled_when_max_is_zero():
    uid = "unlimited-user"
    for _ in range(200):
        check_message_rate_limit(uid, max_messages=0)
    reset_rate_limit(uid)


def test_reset_clears_window():
    uid = "reset-user"
    reset_rate_limit(uid)
    for _ in range(5):
        check_message_rate_limit(uid, max_messages=5)
    reset_rate_limit(uid)
    # Window cleared — should allow again
    check_message_rate_limit(uid, max_messages=5)
    reset_rate_limit(uid)


# ---------------------------------------------------------------------------
# search_backend (via web app)
# ---------------------------------------------------------------------------


class _StubLLM:
    """Deterministic LLM stub for search-flow classification tests."""

    def __init__(self, reply: str) -> None:
        self._reply = reply

    def complete(self, messages, **_):
        return self._reply


def test_search_flow_classification_uses_llm_content(tmp_path, monkeypatch):
    """A configured LLM drives the result from the query content, not a constant."""
    import src.internal.servers.query_and_chat.search_backend as sb

    client, store = _admin_client(tmp_path)

    monkeypatch.setattr(sb, "_build_flow_classifier_llm", lambda: _StubLLM("search"))
    assert (
        client.post(
            "/search/search-flow-classification",
            json={"user_query": "procurement runbook"},
        ).json()["is_search_flow"]
        is True
    )

    monkeypatch.setattr(sb, "_build_flow_classifier_llm", lambda: _StubLLM("chat"))
    assert (
        client.post(
            "/search/search-flow-classification",
            json={"user_query": "write me a poem about spring"},
        ).json()["is_search_flow"]
        is False
    )
    store.close()


def test_search_flow_classification_no_llm_defaults_to_chat(tmp_path, monkeypatch):
    """With no LLM configured, the endpoint defaults to chat instead of guessing."""
    import src.internal.servers.query_and_chat.search_backend as sb

    monkeypatch.setattr(sb, "_build_flow_classifier_llm", lambda: None)
    client, store = _admin_client(tmp_path)
    response = client.post(
        "/search/search-flow-classification",
        json={"user_query": "procurement runbook"},
    )
    assert response.status_code == 200
    assert response.json()["is_search_flow"] is False
    store.close()


def test_search_flow_classification_long_query_is_chat(tmp_path):
    client, store = _admin_client(tmp_path)
    long_query = "x" * 250
    response = client.post(
        "/search/search-flow-classification",
        json={"user_query": long_query},
    )
    assert response.status_code == 200
    assert response.json()["is_search_flow"] is False
    store.close()


def test_send_search_message_non_streaming(tmp_path, monkeypatch):
    async def fake_run_expanded_search(query, **_):
        from src.internal.search.process_search_query import SearchQueryResult
        from src.context.search import SearchResult

        return SearchQueryResult(
            original_query=query,
            executed_queries=[query],
            results=[SearchResult(contents="result", score=0.9, title="T")],
        )

    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.search_backend.run_expanded_search",
        fake_run_expanded_search,
    )
    client, store = _admin_client(tmp_path)
    response = client.post(
        "/search/send-search-message",
        headers=_bearer(_USER_ID),
        json={"search_query": "what is ML?", "stream": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["all_executed_queries"] == ["what is ML?"]
    assert len(data["search_docs"]) == 1
    store.close()


def test_send_search_message_streaming(tmp_path, monkeypatch):
    async def fake_run_expanded_search(query, **_):
        from src.internal.search.process_search_query import SearchQueryResult

        return SearchQueryResult(
            original_query=query,
            executed_queries=[query, "keyword expansion"],
            results=[],
        )

    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.search_backend.run_expanded_search",
        fake_run_expanded_search,
    )
    client, store = _admin_client(tmp_path)
    response = client.post(
        "/search/send-search-message",
        headers=_bearer(_USER_ID),
        json={"search_query": "ML overview", "stream": True},
    )
    assert response.status_code == 200
    lines = [line for line in response.text.strip().split("\n") if line]
    packets = [json.loads(line) for line in lines]
    types = [p["type"] for p in packets]
    assert "search_queries" in types
    assert "search_docs" in types
    store.close()


def test_search_history_requires_auth(tmp_path):
    client, store = _admin_client(tmp_path)
    response = client.get("/search/search-history")
    assert response.status_code == 200
    assert response.json()["search_queries"] == []  # anonymous → empty
    store.close()


def test_search_history_returns_sessions_for_user(tmp_path):
    client, store = _admin_client(tmp_path)
    store.create_chat_session(user_id=_USER_ID, title="My search query")
    response = client.get(
        "/search/search-history",
        headers=_bearer(_USER_ID),
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["search_queries"]) == 1
    assert data["search_queries"][0]["query"] == "My search query"
    store.close()


def test_query_standard_answer_returns_501(tmp_path):
    client, store = _admin_client(tmp_path)
    response = client.get("/query/standard-answer")
    assert response.status_code == 501
    store.close()


# ---------------------------------------------------------------------------
# documents (cc_pair / connector sync)
# ---------------------------------------------------------------------------


def _make_connector(store: AgenticSearchStore, connector_id: str) -> None:
    store.upsert_connector(
        ConnectorConfig(id=connector_id, name="TestConn", source="local_file")
    )


def test_get_connector_last_sync_returns_none_when_no_attempts(tmp_path):
    client, store = _admin_client(tmp_path)
    _make_connector(store, "conn1")
    response = client.get(
        "/manage/admin/connector/conn1/last-sync",
        headers=_bearer(_ADMIN_ID),
    )
    assert response.status_code == 200
    assert response.json() is None
    store.close()


def test_get_connector_last_sync_returns_timestamp_after_attempt(tmp_path):
    client, store = _admin_client(tmp_path)
    _make_connector(store, "conn1")
    store.create_index_attempt(connector_id="conn1")
    response = client.get(
        "/manage/admin/connector/conn1/last-sync",
        headers=_bearer(_ADMIN_ID),
    )
    assert response.status_code == 200
    assert response.json() is not None
    store.close()


def test_trigger_connector_sync_creates_index_attempt(tmp_path):
    client, store = _admin_client(tmp_path)
    _make_connector(store, "conn1")
    response = client.post(
        "/manage/admin/connector/conn1/sync",
        headers=_bearer(_ADMIN_ID),
    )
    assert response.status_code == 200
    assert response.json()["success"] is True
    attempts = store.list_index_attempts(connector_id="conn1")
    assert len(attempts) == 1
    assert attempts[0].status == "not_started"
    store.close()


def test_connector_sync_404_for_missing_connector(tmp_path):
    client, store = _admin_client(tmp_path)
    response = client.post(
        "/manage/admin/connector/nonexistent/sync",
        headers=_bearer(_ADMIN_ID),
    )
    assert response.status_code == 404
    store.close()


def test_connector_sync_requires_admin(tmp_path):
    client, store = _admin_client(tmp_path)
    _make_connector(store, "conn1")
    response = client.post(
        "/manage/admin/connector/conn1/sync",
        headers=_bearer(_USER_ID),
    )
    assert response.status_code == 403
    store.close()


def test_group_sync_returns_501(tmp_path):
    client, store = _admin_client(tmp_path)
    _make_connector(store, "conn1")
    response = client.post(
        "/manage/admin/connector/conn1/sync-groups",
        headers=_bearer(_ADMIN_ID),
    )
    assert response.status_code == 501
    store.close()


def test_group_sync_404_for_missing_connector(tmp_path):
    client, store = _admin_client(tmp_path)
    response = client.post(
        "/manage/admin/connector/nonexistent/sync-groups",
        headers=_bearer(_ADMIN_ID),
    )
    assert response.status_code == 404
    store.close()
