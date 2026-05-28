from __future__ import annotations

from src.db import (
    AgenticSearchStore,
    ConnectorConfig,
    DocumentPermission,
    GroupRecord,
    StoredDocument,
    UserRecord,
)


def test_connector_document_metadata_round_trips(tmp_path):
    db_path = tmp_path / "agentic-search.sqlite3"
    with AgenticSearchStore(db_path) as store:
        connector = store.upsert_connector(
            ConnectorConfig(
                id="conn-local",
                name="Local Files",
                source="local_file",
                config={"root": "/tmp/docs"},
                metadata={"owner": "search"},
            )
        )
        document = store.upsert_document(
            StoredDocument(
                id="doc-1",
                title="Handbook",
                contents="Agentic search handbook",
                url="file:///tmp/docs/handbook.md",
                connector_id=connector.id,
                metadata={"tags": ["handbook"]},
            )
        )

        assert store.get_connector("conn-local") == connector
        assert store.get_document("doc-1") == document
        assert store.list_documents(connector_id="conn-local") == [document]

    with AgenticSearchStore(db_path) as reopened:
        assert reopened.get_document("doc-1") is not None
        assert reopened.get_document("doc-1").metadata == {"tags": ["handbook"]}


def test_users_groups_and_document_permissions(tmp_path):
    with AgenticSearchStore(tmp_path / "state.sqlite3") as store:
        store.upsert_user(UserRecord(id="alice", email="alice@example.test"))
        store.upsert_user(UserRecord(id="bob", email="bob@example.test"))
        store.upsert_group(
            GroupRecord(id="eng", name="Engineering", user_ids=["alice"])
        )

        public_doc = store.upsert_document(
            StoredDocument(id="public", title="Public", contents="For everyone")
        )
        user_doc = store.upsert_document(
            StoredDocument(id="alice-only", title="Alice", contents="For Alice")
        )
        group_doc = store.upsert_document(
            StoredDocument(id="eng-only", title="Engineering", contents="For eng")
        )

        store.grant_document_access(
            DocumentPermission(document_id=public_doc.id, principal_type="public")
        )
        store.grant_document_access(
            DocumentPermission(
                document_id=user_doc.id,
                principal_type="user",
                principal_id="alice",
            )
        )
        store.grant_document_access(
            DocumentPermission(
                document_id=group_doc.id,
                principal_type="group",
                principal_id="eng",
            )
        )

        alice_visible = {doc.id for doc in store.documents_visible_to_user("alice")}
        bob_visible = {doc.id for doc in store.documents_visible_to_user("bob")}
        anonymous_visible = {doc.id for doc in store.documents_visible_to_user(None)}

    assert alice_visible == {"public", "alice-only", "eng-only"}
    assert bob_visible == {"public"}
    assert anonymous_visible == {"public"}


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


def test_index_attempt_lifecycle(tmp_path):
    with AgenticSearchStore(tmp_path / "state.sqlite3") as store:
        store.upsert_connector(
            ConnectorConfig(id="conn", name="Connector", source="fixture")
        )
        attempt = store.create_index_attempt(
            connector_id="conn",
            status="in_progress",
            metadata={"checkpoint": "start"},
        )

        updated = store.update_index_attempt(
            attempt.id,
            status="success",
            total_documents=8,
            total_chunks=21,
            metadata={"checkpoint": "done"},
        )

        assert updated.status == "success"
        assert updated.total_documents == 8
        assert updated.total_chunks == 21
        assert updated.metadata == {"checkpoint": "done"}
        assert store.list_index_attempts(connector_id="conn") == [updated]
