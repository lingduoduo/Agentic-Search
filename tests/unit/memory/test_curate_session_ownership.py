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
    store = AgenticSearchStore(":memory:")
    for uid in {"u1", owner} - {None}:
        store.upsert_user(UserRecord(id=uid))
    session = store.create_chat_session(user_id=owner)
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


def test_a_session_with_no_owner_stays_readable():
    """Ownerless is public, the same rule documented in ``SearchFilters.matches``.

    Anonymous callers share the ``default_user`` bucket and their sessions are
    stored with a NULL ``user_id``, so requiring equality here would silently
    remove ``curate --session-id`` for every anonymous caller.
    """
    store, ownerless = _store_with_session(None)
    assert SECRET in service._gather_sources(store, "default_user", ownerless)
    store.close()
