from __future__ import annotations

from src.internal.db import AgenticSearchStore, UserRecord


def test_chat_session_and_messages_are_stored(tmp_path):
    with AgenticSearchStore(tmp_path / "state.sqlite3") as store:
        store.upsert_user(UserRecord(id="alice"))
        session = store.create_chat_session(
            user_id="alice",
            title="Search support",
            metadata={"channel": "web"},
        )
        first = store.add_chat_message(
            session.id,
            role="user",
            content="Find the deployment doc",
        )
        second = store.add_chat_message(
            session.id,
            role="assistant",
            content="I found one relevant document.",
            metadata={"document_ids": ["doc-1"]},
        )

        assert store.get_chat_session(session.id).metadata == {"channel": "web"}
        assert store.list_chat_messages(session.id) == [first, second]
