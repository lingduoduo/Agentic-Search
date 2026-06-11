"""Tests for HeavyWorker bulk operations."""

from __future__ import annotations

from unittest.mock import patch

from src.internal.db.models import ConnectorConfig, DocumentPermission, StoredDocument
from src.internal.db.store import AgenticSearchStore
from src.internal.servers.backgroundworker.heavy_worker import (
    HeavyWorker,
    PruneConnectorTask,
    SyncDocPermissionsTask,
)


def _make_store(tmp_path):
    store = AgenticSearchStore(tmp_path / "test.sqlite3")
    return store


def test_prune_connector_uses_bulk_delete(tmp_path):
    """prune_connector must call delete_documents_bulk, not delete_document per doc."""
    store = _make_store(tmp_path)
    store.upsert_connector(ConnectorConfig(id="conn-1", name="c", source="test"))
    for i in range(5):
        store.upsert_document(
            StoredDocument(
                id=f"doc-{i}", title="t", contents="c", connector_id="conn-1"
            )
        )
    worker = HeavyWorker(store=store)
    task = PruneConnectorTask(connector_id="conn-1", current_doc_ids={"doc-0"})

    with (
        patch.object(store, "delete_document") as mock_single,
        patch.object(
            store, "delete_documents_bulk", wraps=store.delete_documents_bulk
        ) as mock_bulk,
    ):
        worker.prune_connector(task)

    mock_single.assert_not_called()
    mock_bulk.assert_called_once()
    stale_ids = mock_bulk.call_args[0][0]
    assert set(stale_ids) == {"doc-1", "doc-2", "doc-3", "doc-4"}
    store.close()


def test_sync_doc_permissions_uses_bulk_grant(tmp_path):
    """sync_doc_permissions must call grant_document_access_bulk, not grant_document_access."""
    store = _make_store(tmp_path)
    store.upsert_document(StoredDocument(id="doc-a", title="t", contents="c"))
    store.upsert_document(StoredDocument(id="doc-b", title="t", contents="c"))
    worker = HeavyWorker(store=store)
    task = SyncDocPermissionsTask(
        connector_id="conn-1",
        permissions_by_doc={
            "doc-a": [
                DocumentPermission(
                    document_id="doc-a",
                    principal_type="user",
                    principal_id="alice",
                    access="read",
                ),
                DocumentPermission(
                    document_id="doc-a",
                    principal_type="user",
                    principal_id="bob",
                    access="read",
                ),
            ],
            "doc-b": [
                DocumentPermission(
                    document_id="doc-b", principal_type="public", access="read"
                ),
            ],
        },
    )

    with (
        patch.object(store, "grant_document_access") as mock_single,
        patch.object(
            store, "grant_document_access_bulk", wraps=store.grant_document_access_bulk
        ) as mock_bulk,
    ):
        worker.sync_doc_permissions(task)

    mock_single.assert_not_called()
    mock_bulk.assert_called_once()
    granted_permissions = mock_bulk.call_args[0][0]
    assert len(granted_permissions) == 3
    store.close()


def test_delete_documents_bulk_single_commit(tmp_path):
    """delete_documents_bulk must delete all docs and return correct count."""
    store = _make_store(tmp_path)
    for i in range(5):
        store.upsert_document(StoredDocument(id=f"doc-{i}", title="t", contents="c"))

    deleted = store.delete_documents_bulk([f"doc-{i}" for i in range(5)])

    assert deleted == 5
    assert store.list_documents() == []
    store.close()


def test_grant_document_access_bulk_stores_all_permissions(tmp_path):
    """grant_document_access_bulk must store all permissions in one shot."""
    store = _make_store(tmp_path)
    store.upsert_document(StoredDocument(id="doc-x", title="t", contents="c"))

    permissions = [
        DocumentPermission(
            document_id="doc-x",
            principal_type="user",
            principal_id=f"u{i}",
            access="read",
        )
        for i in range(10)
    ]
    count = store.grant_document_access_bulk(permissions)

    assert count == 10
    stored = store.get_document_permissions("doc-x")
    assert len(stored) == 10
    store.close()
