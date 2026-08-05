"""``curate --session-id`` may only read sessions the caller can already reach.

Without an ownership check, naming another user's session hands their whole
transcript to the LLM and files the distilled result in the caller's own
memories -- a cross-user read of conversation content. The unfiltered branch
(`list_sessions_for_user`) has always been scoped; only the by-id branch was not.
"""

from __future__ import annotations

import asyncio

from src.internal.db.models import UserRecord
from src.internal.db.store import AgenticSearchStore
from src.internal.memory import service

SECRET = "My bank PIN is 4242."


class _ExplodingLLM:
    """Any call means the transcript reached the model. That is the leak."""

    config = type("cfg", (), {"model_name": "fake-model"})()

    def stream(self, *args, **kwargs):
        raise AssertionError("the LLM was handed a transcript it may not read")


def _store_with_session(owner: str | None) -> tuple[AgenticSearchStore, str]:
    """*owner* of None means a legacy NULL row, which the API no longer creates.

    Anonymous sessions carry ANONYMOUS_USER_ID now, so a genuinely ownerless row
    only arises from data written before that or from ON DELETE SET NULL. It is
    forced here with SQL for exactly that reason.
    """
    store = AgenticSearchStore(":memory:")
    for uid in {"u1", owner} - {None}:
        store.upsert_user(UserRecord(id=uid))
    session = store.create_chat_session(user_id=owner)
    if owner is None:
        store._conn.execute(
            "UPDATE chat_sessions SET user_id = NULL WHERE id = ?", (session.id,)
        )
        store._conn.commit()
    store.add_chat_message(session.id, role="user", content=SECRET)
    return store, session.id


def _curate(store, user_id: str, session_id: str) -> dict:
    return asyncio.run(
        service.curate_from_conversation(
            store, user_id, _ExplodingLLM(), session_id=session_id
        )
    )


def test_another_users_session_is_not_read():
    store, theirs = _store_with_session("u2")
    summary = _curate(store, "u1", theirs)
    assert summary["status"] == "empty"
    store.close()


def test_the_callers_own_session_is_still_read():
    """The control: proves the check withholds rather than blanket-refusing."""
    store, mine = _store_with_session("u1")
    sources = service._gather_sources(store, "u1", mine)
    assert SECRET in sources
    store.close()


def test_a_session_with_no_owner_is_not_read_either():
    """Ownership is strict: only a session whose ``user_id`` matches is read.

    Ownerless sessions were readable by anyone who knew the id, on the same
    "declares no ACL means public" rule documents follow. Sessions are not
    documents — an ownerless one is somebody's actual conversation, just one
    recorded before they signed in — so the rule no longer carries over.
    """
    store, ownerless = _store_with_session(None)
    assert service._gather_sources(store, "default_user", ownerless) == ""
    store.close()


def test_an_unreadable_session_id_says_so_instead_of_looking_empty():
    """The capability loss must be visible, not silent.

    Anonymous callers' sessions are stored with a NULL ``user_id``, so strict
    ownership removes ``curate --session-id`` for all of them. Reusing the
    generic "no conversations or notes yet" would make that read as "nothing to
    do" -- the invisible-loss shape that #490 shipped and #491 had to undo.

    One message covers both causes (not yours, does not exist), so it still
    confirms nothing about anyone else's session.
    """
    store, ownerless = _store_with_session(None)
    summary = _curate(store, "default_user", ownerless)
    assert summary["status"] == "empty"
    assert summary["message"] == "session not found, or not readable by you"
    store.close()


def test_no_session_id_keeps_the_generic_empty_message():
    """The control: the new message is scoped to an explicit session id."""
    store = AgenticSearchStore(":memory:")
    summary = asyncio.run(
        service.curate_from_conversation(store, "nobody", _ExplodingLLM())
    )
    assert summary["message"] == "no conversations or notes yet"
    store.close()
