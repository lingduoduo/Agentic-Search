"""Anonymous sessions carry the anonymous identity, not a NULL owner.

#488 settled that anonymous is an identity rather than the absence of one — it
carries ``["public"]``. Sessions did not follow: they were stored with
``user_id = NULL``, because ``chat_sessions.user_id`` has a foreign key to
``users(id)`` and no row existed to point at, while ``user_memories.user_id``
has no such constraint and so worked fine with the same id.

That split is why an anonymous caller's memories and their conversations lived
under different keys, and why strict session ownership left them with nothing to
curate.
"""

from __future__ import annotations

from src.internal.db.models import ANONYMOUS_USER_ID, UserRecord
from src.internal.db.store import AgenticSearchStore
from src.internal.memory import service


def test_a_session_created_without_a_user_is_owned_by_the_anonymous_identity():
    store = AgenticSearchStore(":memory:")
    session = store.create_chat_session()
    assert session.user_id == ANONYMOUS_USER_ID
    store.close()


def test_the_anonymous_identity_exists_so_the_foreign_key_holds():
    """Without a provisioned row this raises IntegrityError, which is the whole
    reason anonymous sessions were NULL in the first place."""
    store = AgenticSearchStore(":memory:")
    assert store.get_user(ANONYMOUS_USER_ID) is not None
    store.close()


def test_the_anonymous_session_owner_is_the_anonymous_memory_bucket():
    """The two must be the same id, or curate reads a different bucket than it
    writes and silently finds nothing."""
    assert service.DEFAULT_MEMORY_USER_ID == ANONYMOUS_USER_ID


def test_an_anonymous_caller_can_curate_from_their_own_conversations_again():
    store = AgenticSearchStore(":memory:")
    session = store.create_chat_session()
    store.add_chat_message(session.id, role="user", content="I moved to Shanghai.")

    # Both routes: the no-flag path and the explicit session id.
    assert [s.id for s in store.list_sessions_for_user(ANONYMOUS_USER_ID)] == [
        session.id
    ]
    assert "Shanghai" in service._gather_sources(store, ANONYMOUS_USER_ID, session.id)
    store.close()


def test_an_explicit_user_still_owns_their_own_sessions():
    """The control: the default applies only when no user was given."""
    store = AgenticSearchStore(":memory:")
    store.upsert_user(UserRecord(id="alice"))
    session = store.create_chat_session(user_id="alice")
    assert session.user_id == "alice"
    assert service._gather_sources(store, ANONYMOUS_USER_ID, session.id) == ""
    store.close()


def test_pre_existing_null_sessions_are_left_alone():
    """No backfill, deliberately.

    ``chat_sessions.user_id`` is ``ON DELETE SET NULL``, so a NULL row is either
    a legacy anonymous session *or* an orphaned session whose owner was deleted.
    Adopting them into the shared anonymous bucket would hand a deleted user's
    conversations to every anonymous caller.
    """
    store = AgenticSearchStore(":memory:")
    session = store.create_chat_session()
    store._conn.execute(
        "UPDATE chat_sessions SET user_id = NULL WHERE id = ?", (session.id,)
    )
    store._conn.commit()

    assert store.list_sessions_for_user(ANONYMOUS_USER_ID) == []
    assert service._gather_sources(store, ANONYMOUS_USER_ID, session.id) == ""
    store.close()


# --- The anonymous row must stay invisible to account management ---------------
#
# It exists only to satisfy the chat_sessions foreign key. Anywhere the product
# counts or lists *accounts*, it is not one, and letting it leak there broke
# things no test was watching.


def test_the_first_real_registrant_still_becomes_admin():
    """The regression that provisioning the row introduced.

    `/auth/register` grants admin with `role = "admin" if not all_users`. A
    permanent anonymous row makes that list never empty, so a fresh deployment
    would end up with no admin at all -- silently, since registration still
    returns 200.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.internal.configs.app_configs import load_app_settings
    from src.internal.servers.users.api import create_users_router

    store = AgenticSearchStore(":memory:")
    app = FastAPI()
    app.include_router(create_users_router(store, load_app_settings()))

    first = TestClient(app).post(
        "/auth/register",
        json={"email": "first@localhost", "username": "first", "password": "pw"},
    )
    assert first.json()["role"] == "admin"
    store.close()


def test_list_users_omits_the_anonymous_identity():
    store = AgenticSearchStore(":memory:")
    store.upsert_user(UserRecord(id="alice"))
    assert [u.id for u in store.list_users()] == ["alice"]
    store.close()


def test_the_row_is_still_there_for_the_foreign_key():
    """Hidden from listings, not absent: sessions reference it."""
    store = AgenticSearchStore(":memory:")
    assert store.get_user(ANONYMOUS_USER_ID) is not None
    assert store.create_chat_session().user_id == ANONYMOUS_USER_ID
    store.close()
