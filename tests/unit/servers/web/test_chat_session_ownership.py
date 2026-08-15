"""The /chat/* session endpoints enforce ownership.

`/chat/*` duplicates the `/api/sessions` surface, and shipped with the same
missing ownership check. #532 fixed `/api/sessions/{id}`; this copy stayed live,
because the guard lived in only one of the two routers.

It was worse here than there. `/api/sessions` leaked; `/chat/*` also **renamed
and permanently deleted** other users' sessions:

    GET    /chat/get-chat-session/{id}    (no creds) -> 200, full transcript
    PUT    /chat/rename-chat-session      (no creds) -> 200, title overwritten
    DELETE /chat/delete-chat-session/{id} (no creds) -> 200, session gone

The predicate now lives in `servers/_auth.py` so a third copy of these endpoints
cannot be written without it.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.internal.db.models import UserRecord
from src.internal.db.store import AgenticSearchStore
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


@pytest.fixture()
def client_and_store(monkeypatch):
    # The repo's .env sets AGENTIC_SEARCH_DEV_ADMIN=1 and dotenv overrides
    # `env -u`, so it must be neutralised explicitly or these pass against an
    # unguarded app.
    monkeypatch.setitem(os.environ, "AGENTIC_SEARCH_DEV_ADMIN", "false")
    directory = Path(tempfile.mkdtemp())
    db_path = directory / "chat.sqlite3"
    app = create_web_app(SearchExperienceSettings(db_path=db_path))
    store = AgenticSearchStore(str(db_path))
    with TestClient(app) as client:
        yield client, store


def _owned(store: AgenticSearchStore) -> str:
    owner = store.upsert_user(UserRecord(id="user_alice", email="alice@example.com"))
    session = store.create_chat_session(user_id=owner.id, title="ALICE-SECRET")
    store.add_chat_message(session.id, role="user", content="ALICE-SECRET-BODY")
    return session.id


def test_reading_another_users_chat_session_is_refused(client_and_store):
    client, store = client_and_store
    session_id = _owned(store)

    response = client.get(f"/chat/get-chat-session/{session_id}")

    assert response.status_code == 404
    assert "ALICE-SECRET" not in response.text


def test_renaming_another_users_chat_session_is_refused(client_and_store):
    client, store = client_and_store
    session_id = _owned(store)

    response = client.put(
        "/chat/rename-chat-session",
        json={"chat_session_id": session_id, "name": "PWNED"},
    )

    assert response.status_code == 404
    assert store.get_chat_session(session_id).title == "ALICE-SECRET"


def test_deleting_another_users_chat_session_is_refused(client_and_store):
    """The destructive one, and the reason this could not wait.

    Unlike a disclosure bug, this had no recovery: the session was gone.
    """
    client, store = client_and_store
    session_id = _owned(store)

    response = client.delete(f"/chat/delete-chat-session/{session_id}")

    assert response.status_code == 404
    assert store.get_chat_session(session_id) is not None


def test_anonymous_sessions_remain_usable_by_id(client_and_store):
    """Deliberate narrowing, not an oversight.

    Signed-out callers have no identity to compare against, so the id is the
    only capability there is. Guarding these would break the signed-out flows
    the surface exists to serve. Per-caller anonymous identity is separate,
    unbuilt work.
    """
    client, store = client_and_store
    anonymous = store.create_chat_session(user_id=None, title="anon")

    assert client.get(f"/chat/get-chat-session/{anonymous.id}").status_code == 200
    assert client.delete(f"/chat/delete-chat-session/{anonymous.id}").status_code == 200


def test_the_guard_is_shared_between_both_session_surfaces(client_and_store):
    """Why a third copy of these endpoints cannot silently ship unguarded.

    `/api/sessions` and `/chat/*` cannot import each other, which is how the
    check ended up in only one. It now lives in `servers/_auth.py`, and both
    import it from there.
    """
    from src.internal.servers._auth import caller_may_use_session
    from src.internal.servers.query_and_chat import chat_backend
    from src.internal.servers.web import app as web_app

    assert chat_backend.caller_may_use_session is caller_may_use_session
    assert web_app.caller_may_use_session is caller_may_use_session
