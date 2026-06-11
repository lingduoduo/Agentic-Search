# Optimize Document Indexing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate N+1 SQLite write patterns in the document indexing pipeline and reduce OpenSearch round-trips in hot paths, cutting per-batch commit overhead and query latency.

**Architecture:** Five targeted changes across three layers — (1) add batch write methods to `AgenticSearchStore`; (2) update `DocprocessingWorker` and `HeavyWorker` to call them instead of looping; (3) add `msearch` to `OpenSearchIndexClient` and batch `id_based_retrieval`; (4) cache the `OpenAI` HTTP client inside `OpenAIEmbedder`. Each change is independently verifiable.

**Tech Stack:** Python 3.11+, sqlite3 (stdlib), opensearch-py, pytest, unittest.mock.

---

## File Map

| Action | File | Change |
|--------|------|--------|
| Modify | `src/internal/db/store.py` | Add `upsert_documents_bulk()`, `delete_documents_bulk()`, `grant_document_access_bulk()` |
| Modify | `src/internal/servers/backgroundworker/docprocessing.py` | Use `upsert_documents_bulk()` in `process_batch()` |
| Modify | `src/internal/servers/backgroundworker/heavy_worker.py` | Use `delete_documents_bulk()` in `prune_connector()`, `grant_document_access_bulk()` in `sync_doc_permissions()` |
| Modify | `src/internal/document_index/opensearch/client.py` | Add `msearch()` to `OpenSearchIndexClient` |
| Modify | `src/internal/document_index/opensearch/opensearch_document_index.py` | Use `msearch()` in `id_based_retrieval()` |
| Modify | `src/internal/document_index/embedding_cache.py` | Cache `OpenAI` client in `OpenAIEmbedder.__init__()` |
| Modify | `tests/unit/test_db_store.py` | Tests for `upsert_documents_bulk`, `delete_documents_bulk`, `grant_document_access_bulk` |
| Create | `tests/unit/servers/backgroundworker/test_heavy_worker.py` | Tests for bulk prune and bulk ACL sync |
| Create | `tests/unit/servers/backgroundworker/test_docprocessing.py` | Tests for bulk upsert in `process_batch()` |
| Modify | `tests/unit/test_embedding_cache.py` | Test that `OpenAI` client is reused across embed() calls |

---

## Task 1: Bulk document upsert in `AgenticSearchStore`

**Problem:** `process_batch()` in `docprocessing.py:110-122` calls `upsert_document()` once per document, each of which calls `self._conn.commit()`. A batch of 100 documents triggers 100 separate SQLite write transactions.

**Files:**
- Modify: `src/internal/db/store.py` — add `upsert_documents_bulk()`
- Modify: `tests/unit/test_db_store.py` — add test

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_db_store.py`:

```python
def test_upsert_documents_bulk_single_transaction(tmp_path, monkeypatch):
    """upsert_documents_bulk must commit exactly once, not once per document."""
    db_path = tmp_path / "bulk.sqlite3"
    with AgenticSearchStore(db_path) as store:
        commit_calls = []
        real_commit = store._conn.commit
        monkeypatch.setattr(store._conn, "commit", lambda: (commit_calls.append(1), real_commit())[1])

        docs = [
            StoredDocument(id=f"doc-{i}", title=f"T{i}", contents=f"C{i}")
            for i in range(10)
        ]
        result = store.upsert_documents_bulk(docs)

        assert len(result) == 10
        assert commit_calls == [1]  # exactly one commit
        for doc in result:
            assert store.get_document(doc.id) is not None


def test_upsert_documents_bulk_preserves_created_at(tmp_path):
    """Re-upserting via bulk must not overwrite created_at for existing docs."""
    db_path = tmp_path / "bulk_ts.sqlite3"
    with AgenticSearchStore(db_path) as store:
        original = store.upsert_document(
            StoredDocument(id="doc-1", title="Old", contents="old content")
        )
        original_created_at = store.get_document("doc-1").created_at

        store.upsert_documents_bulk([
            StoredDocument(id="doc-1", title="New", contents="new content")
        ])

        updated = store.get_document("doc-1")
        assert updated.title == "New"
        assert updated.created_at == original_created_at
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_db_store.py::test_upsert_documents_bulk_single_transaction tests/unit/test_db_store.py::test_upsert_documents_bulk_preserves_created_at -v
```

Expected: FAIL with `AttributeError: 'AgenticSearchStore' object has no attribute 'upsert_documents_bulk'`

- [ ] **Step 3: Implement `upsert_documents_bulk()` in `store.py`**

Add after the `upsert_document()` method (around line 413):

```python
def upsert_documents_bulk(self, documents: list[StoredDocument]) -> list[StoredDocument]:
    """Upsert multiple documents in a single transaction."""
    if not documents:
        return []
    now = _now()
    doc_ids = [d.id for d in documents]
    placeholders = ", ".join("?" * len(doc_ids))
    existing_created_at: dict[str, str] = {
        row["id"]: str(row["created_at"])
        for row in self._conn.execute(
            f"SELECT id, created_at FROM documents WHERE id IN ({placeholders})",
            tuple(doc_ids),
        ).fetchall()
    }
    records: list[StoredDocument] = []
    params: list[tuple] = []
    for document in documents:
        created_at = existing_created_at.get(document.id, now)
        record = StoredDocument(
            id=document.id,
            title=document.title,
            contents=document.contents,
            url=document.url,
            connector_id=document.connector_id,
            metadata=dict(document.metadata),
            created_at=created_at,
            updated_at=now,
        )
        records.append(record)
        params.append((
            record.id,
            record.title,
            record.contents,
            record.url,
            record.connector_id,
            _json_dumps(record.metadata),
            record.created_at,
            record.updated_at,
        ))
    self._conn.executemany(
        """
        INSERT INTO documents (
            id, title, contents, url, connector_id, metadata_json, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title = excluded.title,
            contents = excluded.contents,
            url = excluded.url,
            connector_id = excluded.connector_id,
            metadata_json = excluded.metadata_json,
            updated_at = excluded.updated_at
        """,
        params,
    )
    self._conn.commit()
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_db_store.py::test_upsert_documents_bulk_single_transaction tests/unit/test_db_store.py::test_upsert_documents_bulk_preserves_created_at -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/internal/db/store.py tests/unit/test_db_store.py
git commit -m "perf: add upsert_documents_bulk to AgenticSearchStore"
```

---

## Task 2: Use `upsert_documents_bulk()` in `DocprocessingWorker`

**Problem:** `process_batch()` at `docprocessing.py:110-122` and `:138-155` calls `upsert_document()` in a per-document loop. With 100 docs per batch this is 200 individual commits (one for initial persistence, one for metadata update).

**Files:**
- Modify: `src/internal/servers/backgroundworker/docprocessing.py`
- Create: `tests/unit/servers/backgroundworker/test_docprocessing.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/servers/backgroundworker/test_docprocessing.py`:

```python
"""Tests for DocprocessingWorker bulk path."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import numpy as np

from src.internal.connectors.models import Document
from src.internal.db.models import StoredDocument
from src.internal.db.store import AgenticSearchStore
from src.internal.servers.backgroundworker.docprocessing import (
    DocprocessingWorker,
    DocprocessingConfig,
)
from src.internal.document_index import DefaultIndexingEmbedder
from src.internal.document_index.models import ChunkingConfig, EmbeddingConfig


def _make_docs(n: int) -> list[Document]:
    return [Document(id=f"doc-{i}", title=f"T{i}", contents=f"Content {i}") for i in range(n)]


def test_process_batch_uses_bulk_upsert(tmp_path):
    """process_batch should call upsert_documents_bulk once, not upsert_document N times."""
    store = AgenticSearchStore(tmp_path / "test.sqlite3")
    worker = DocprocessingWorker(
        store=store,
        chunk_sink=None,
        embedder=DefaultIndexingEmbedder(
            embedding_fn=lambda texts: np.ones((len(texts), 4), dtype=np.float32),
            config=EmbeddingConfig(retrieval_method="contriever"),
        ),
        config=DocprocessingConfig(chunking=ChunkingConfig(include_title=False, include_metadata=False)),
    )
    docs = _make_docs(5)

    with patch.object(store, "upsert_document") as mock_single, \
         patch.object(store, "upsert_documents_bulk", wraps=store.upsert_documents_bulk) as mock_bulk:
        worker.process_batch(docs)

    mock_single.assert_not_called()
    assert mock_bulk.call_count >= 1  # at least initial upsert + metadata update
    store.close()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/servers/backgroundworker/test_docprocessing.py::test_process_batch_uses_bulk_upsert -v
```

Expected: FAIL (`upsert_document` is called, `upsert_documents_bulk` is not called)

- [ ] **Step 3: Update `process_batch()` in `docprocessing.py`**

Replace `docprocessing.py:110-155` (both upsert loops) with bulk calls:

```python
# Step 1: SQL upsert — persist raw documents before any processing
if self._store is not None:
    self._store.upsert_documents_bulk([
        StoredDocument(
            id=doc.id,
            title=doc.title or "",
            contents=doc.contents or "",
            url=doc.url,
            connector_id=self._config.connector_id,
            metadata=dict(doc.metadata or {}),
        )
        for doc in documents
    ])

# 2-5. Filter, chunk, embed, and write to the configured index.
indexing_result = index_documents(
    documents,
    sink=self._sink,
    chunking=self._config.chunking,
    embedding=self._config.embedding,
    embedder=self._embedder,
    callback=self._callback,
    retry_sleep_seconds=self._config.vector_db_retry_sleep_secs,
)
result.total_chunks = len(indexing_result.chunks)
result.failures.extend(indexing_result.failures)

# 6. Metadata update — record chunk count on successfully indexed docs.
if self._store is not None:
    chunks_by_doc = indexing_result.successful_chunk_counts
    metadata_updates = [
        StoredDocument(
            id=doc.id,
            title=doc.title or "",
            contents=doc.contents or "",
            url=doc.url,
            connector_id=self._config.connector_id,
            metadata={**(doc.metadata or {}), "indexed_chunks": chunks_by_doc[doc.id]},
        )
        for doc in indexing_result.documents
        if doc.id in chunks_by_doc
    ]
    if metadata_updates:
        self._store.upsert_documents_bulk(metadata_updates)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/servers/backgroundworker/test_docprocessing.py::test_process_batch_uses_bulk_upsert -v
```

Expected: PASS

- [ ] **Step 5: Run the full test suite**

```bash
pytest tests/unit/ -v --tb=short -q
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/internal/servers/backgroundworker/docprocessing.py tests/unit/servers/backgroundworker/test_docprocessing.py
git commit -m "perf: use upsert_documents_bulk in DocprocessingWorker.process_batch"
```

---

## Task 3: Bulk delete and bulk ACL write in `AgenticSearchStore` + `HeavyWorker`

**Problem:**
- `prune_connector()` at `heavy_worker.py:174-185` loops over stale doc IDs calling `delete_document()` one per commit.
- `sync_doc_permissions()` at `heavy_worker.py:187-199` has a nested loop calling `grant_document_access()` once per permission, each committing individually.

**Files:**
- Modify: `src/internal/db/store.py` — add `delete_documents_bulk()`, `grant_document_access_bulk()`
- Modify: `src/internal/servers/backgroundworker/heavy_worker.py`
- Create: `tests/unit/servers/backgroundworker/test_heavy_worker.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/servers/backgroundworker/test_heavy_worker.py`:

```python
"""Tests for HeavyWorker bulk operations."""
from __future__ import annotations

from unittest.mock import patch

from src.internal.db.models import DocumentPermission, StoredDocument
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
    for i in range(5):
        store.upsert_document(StoredDocument(id=f"doc-{i}", title="t", contents="c"))
    worker = HeavyWorker(store=store)
    task = PruneConnectorTask(connector_id="conn-1", current_doc_ids={"doc-0"})

    with patch.object(store, "delete_document") as mock_single, \
         patch.object(store, "delete_documents_bulk", wraps=store.delete_documents_bulk) as mock_bulk:
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
                DocumentPermission(document_id="doc-a", principal_type="user", principal_id="alice", access="read"),
                DocumentPermission(document_id="doc-a", principal_type="user", principal_id="bob", access="read"),
            ],
            "doc-b": [
                DocumentPermission(document_id="doc-b", principal_type="public", access="read"),
            ],
        },
    )

    with patch.object(store, "grant_document_access") as mock_single, \
         patch.object(store, "grant_document_access_bulk", wraps=store.grant_document_access_bulk) as mock_bulk:
        worker.sync_doc_permissions(task)

    mock_single.assert_not_called()
    mock_bulk.assert_called_once()
    granted_permissions = mock_bulk.call_args[0][0]
    assert len(granted_permissions) == 3
    store.close()


def test_delete_documents_bulk_single_commit(tmp_path, monkeypatch):
    """delete_documents_bulk must issue one DELETE and one commit."""
    store = _make_store(tmp_path)
    for i in range(5):
        store.upsert_document(StoredDocument(id=f"doc-{i}", title="t", contents="c"))

    commit_calls = []
    real_commit = store._conn.commit
    monkeypatch.setattr(store._conn, "commit", lambda: (commit_calls.append(1), real_commit())[1])

    deleted = store.delete_documents_bulk([f"doc-{i}" for i in range(5)])

    assert deleted == 5
    assert commit_calls == [1]
    assert store.list_documents() == []
    store.close()


def test_grant_document_access_bulk_single_commit(tmp_path, monkeypatch):
    """grant_document_access_bulk must issue one executemany and one commit."""
    store = _make_store(tmp_path)
    store.upsert_document(StoredDocument(id="doc-x", title="t", contents="c"))

    commit_calls = []
    real_commit = store._conn.commit
    monkeypatch.setattr(store._conn, "commit", lambda: (commit_calls.append(1), real_commit())[1])

    permissions = [
        DocumentPermission(document_id="doc-x", principal_type="user", principal_id=f"u{i}", access="read")
        for i in range(10)
    ]
    count = store.grant_document_access_bulk(permissions)

    assert count == 10
    assert commit_calls == [1]
    stored = store.get_document_permissions("doc-x")
    assert len(stored) == 10
    store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/servers/backgroundworker/test_heavy_worker.py -v
```

Expected: FAIL with `AttributeError: 'AgenticSearchStore' object has no attribute 'delete_documents_bulk'`

- [ ] **Step 3: Add `delete_documents_bulk()` and `grant_document_access_bulk()` to `store.py`**

Add after `delete_document()` (around line 441):

```python
def delete_documents_bulk(self, doc_ids: list[str]) -> int:
    """Delete multiple documents by ID in a single transaction. Returns count deleted."""
    if not doc_ids:
        return 0
    placeholders = ", ".join("?" * len(doc_ids))
    cursor = self._conn.execute(
        f"DELETE FROM documents WHERE id IN ({placeholders})",
        tuple(doc_ids),
    )
    self._conn.commit()
    return cursor.rowcount
```

Add after `grant_document_access()` (around line 609):

```python
def grant_document_access_bulk(self, permissions: list[DocumentPermission]) -> int:
    """Write multiple permission grants in a single transaction. Returns count written."""
    if not permissions:
        return 0
    now = _now()
    params = [
        (
            p.document_id,
            p.principal_type,
            p.principal_id or "",
            p.access,
            p.created_at or now,
        )
        for p in permissions
    ]
    self._conn.executemany(
        """
        INSERT OR REPLACE INTO document_permissions (
            document_id, principal_type, principal_id, access, created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        params,
    )
    self._conn.commit()
    return len(params)
```

- [ ] **Step 4: Update `prune_connector()` in `heavy_worker.py`**

Replace `heavy_worker.py:174-185`:

```python
def prune_connector(self, task: PruneConnectorTask) -> None:
    """Delete stored documents that are absent from *current_doc_ids*."""
    stored = self._store.list_documents(connector_id=task.connector_id)
    stale = [doc.id for doc in stored if doc.id not in task.current_doc_ids]
    self._store.delete_documents_bulk(stale)
    logger.info(
        "prune_connector: connector=%r stale=%d total=%d",
        task.connector_id,
        len(stale),
        len(stored),
    )
```

- [ ] **Step 5: Update `sync_doc_permissions()` in `heavy_worker.py`**

Replace `heavy_worker.py:187-199`:

```python
def sync_doc_permissions(self, task: SyncDocPermissionsTask) -> None:
    """Overwrite ACL entries for documents in *permissions_by_doc*."""
    all_permissions = [p for perms in task.permissions_by_doc.values() for p in perms]
    total = self._store.grant_document_access_bulk(all_permissions)
    logger.info(
        "sync_doc_permissions: connector=%r docs=%d entries=%d",
        task.connector_id,
        len(task.permissions_by_doc),
        total,
    )
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/unit/servers/backgroundworker/test_heavy_worker.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 7: Run the full test suite**

```bash
pytest tests/unit/ -v --tb=short -q
```

Expected: all tests pass

- [ ] **Step 8: Commit**

```bash
git add src/internal/db/store.py src/internal/servers/backgroundworker/heavy_worker.py tests/unit/servers/backgroundworker/test_heavy_worker.py
git commit -m "perf: bulk delete and ACL writes in HeavyWorker; add store bulk methods"
```

---

## Task 4: Multi-search for `id_based_retrieval()`

**Problem:** `id_based_retrieval()` at `opensearch_document_index.py:686-720` issues one OpenSearch search per `DocumentSectionRequest`. For N requests this is N sequential HTTP round-trips. OpenSearch's `msearch` API batches N queries into a single HTTP call.

**Files:**
- Modify: `src/internal/document_index/opensearch/client.py` — add `msearch()` to `OpenSearchIndexClient`
- Modify: `src/internal/document_index/opensearch/opensearch_document_index.py` — update `id_based_retrieval()`
- Create: `tests/unit/document_index/test_opensearch_client.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/document_index/test_opensearch_client.py`:

```python
"""Tests for OpenSearchIndexClient.msearch."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_hit(doc_id: str, chunk_index: int) -> dict:
    return {
        "_id": f"{doc_id}_{chunk_index}",
        "_score": 1.0,
        "_source": {
            "document_id": doc_id,
            "chunk_index": chunk_index,
            "blurb": "blurb",
            "content": "content",
            "source_type": "web",
            "semantic_identifier": "test",
            "title": "Test Doc",
            "global_boost": 0,
            "hidden": False,
            "last_updated": None,
            "public": True,
            "access_control_list": [],
            "metadata_list": None,
            "metadata_suffix": "",
            "source_links": None,
            "image_file_id": None,
            "doc_summary": None,
            "chunk_context": None,
            "document_sets": None,
            "user_projects": None,
            "personas": None,
            "primary_owners": None,
            "secondary_owners": None,
            "tenant_id": None,
            "ancestor_hierarchy_node_ids": None,
        },
    }


def test_msearch_issues_single_http_call():
    """msearch must call self._client.msearch once regardless of query count."""
    with patch("opensearchpy.OpenSearch") as MockOS:
        mock_client = MagicMock()
        MockOS.return_value = mock_client
        mock_client.msearch.return_value = {
            "responses": [
                {"hits": {"hits": [_make_hit("doc-1", 0)]}},
                {"hits": {"hits": [_make_hit("doc-2", 0), _make_hit("doc-2", 1)]}},
            ]
        }

        from src.internal.document_index.opensearch.client import OpenSearchIndexClient
        client = OpenSearchIndexClient(index_name="test-index")

        queries = [{"query": {"term": {"document_id": "doc-1"}}},
                   {"query": {"term": {"document_id": "doc-2"}}}]
        result = client.msearch(queries)

    mock_client.msearch.assert_called_once()
    assert len(result) == 2
    assert len(result[0]) == 1
    assert len(result[1]) == 2


def test_msearch_returns_empty_for_empty_queries():
    """msearch with no queries must not hit OpenSearch at all."""
    with patch("opensearchpy.OpenSearch") as MockOS:
        mock_client = MagicMock()
        MockOS.return_value = mock_client

        from src.internal.document_index.opensearch.client import OpenSearchIndexClient
        client = OpenSearchIndexClient(index_name="test-index")
        result = client.msearch([])

    mock_client.msearch.assert_not_called()
    assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/document_index/test_opensearch_client.py -v
```

Expected: FAIL with `AttributeError: 'OpenSearchIndexClient' object has no attribute 'msearch'`

- [ ] **Step 3: Add `msearch()` to `OpenSearchIndexClient` in `client.py`**

Add after the `search()` method (around line 1376):

```python
@log_function_time(print_only=True, debug_only=True)
def msearch(
    self,
    queries: list[dict[str, Any]],
) -> list[list[SearchHit[DocumentChunkWithoutVectors]]]:
    """Execute multiple search queries in a single HTTP call via OpenSearch msearch API.

    Args:
        queries: List of query bodies (same format as the body arg to ``search()``).
            Each query runs against ``self._index_name``.

    Returns:
        One list of SearchHit per input query, in the same order.
    """
    if not queries:
        return []

    # Build NDJSON body: alternating header + query dicts
    body: list[dict[str, Any]] = []
    for query_body in queries:
        body.append({"index": self._index_name})
        body.append(query_body)

    result: dict[str, Any] = self._client.msearch(body=body)
    responses: list[dict[str, Any]] = result.get("responses", [])

    all_hits: list[list[SearchHit[DocumentChunkWithoutVectors]]] = []
    for response in responses:
        hits_raw: list[Any] = response.get("hits", {}).get("hits", [])
        hits: list[SearchHit[DocumentChunkWithoutVectors]] = []
        for hit in hits_raw:
            source: dict[str, Any] | None = hit.get("_source")
            if not source:
                raise RuntimeError(
                    f'Document chunk with ID "{hit.get("_id", "")}" has no data.'
                )
            hits.append(
                SearchHit[DocumentChunkWithoutVectors](
                    document_chunk=DocumentChunkWithoutVectors.model_validate(source),
                    score=hit.get("_score"),
                    match_highlights=hit.get("highlight", {}),
                )
            )
        all_hits.append(hits)
    return all_hits
```

- [ ] **Step 4: Update `id_based_retrieval()` in `opensearch_document_index.py`**

Replace `id_based_retrieval()` method body (lines 686-720):

```python
def id_based_retrieval(
    self,
    chunk_requests: list[DocumentSectionRequest],
    filters: IndexFilters,
    batch_retrieval: bool = False,  # noqa: ARG002
) -> list[InferenceChunk]:
    logger.debug(
        "[OpenSearchDocumentIndex] Retrieving %s chunks for index %s.",
        len(chunk_requests),
        self._index_name,
    )
    if not chunk_requests:
        return []

    # Build one query per request.
    queries = [
        DocumentQuery.get_from_document_id_query(
            document_id=chunk_request.document_id,
            tenant_state=self._tenant_state,
            index_filters=filters,
            include_hidden=False,
            max_chunk_size=chunk_request.max_chunk_size,
            min_chunk_index=chunk_request.min_chunk_ind,
            max_chunk_index=chunk_request.max_chunk_ind,
        )
        for chunk_request in chunk_requests
    ]

    # Fire all queries in one HTTP round-trip.
    all_hit_lists = self._client.msearch(queries)

    results: list[InferenceChunk] = []
    for hit_list in all_hit_lists:
        uncleaned: list[InferenceChunkUncleaned] = [
            _convert_retrieved_opensearch_chunk_to_inference_chunk_uncleaned(
                hit.document_chunk, None, {}
            )
            for hit in hit_list
        ]
        results.extend(cleanup_content_for_chunks(uncleaned))
    return results
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/document_index/test_opensearch_client.py -v
```

Expected: both tests PASS

- [ ] **Step 6: Run the full test suite**

```bash
pytest tests/unit/ -v --tb=short -q
```

Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add src/internal/document_index/opensearch/client.py src/internal/document_index/opensearch/opensearch_document_index.py tests/unit/document_index/test_opensearch_client.py
git commit -m "perf: use msearch in id_based_retrieval to reduce OpenSearch round-trips"
```

---

## Task 5: Cache OpenAI client in `OpenAIEmbedder`

**Problem:** `_call_openai()` at `embedding_cache.py:178` creates a `new OpenAI(...)` instance on every call. Under load, each miss batch creates a new HTTP connection pool (httpx transport) that is never reused, increasing latency and memory pressure.

**Files:**
- Modify: `src/internal/document_index/embedding_cache.py`
- Modify: `tests/unit/test_embedding_cache.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_embedding_cache.py`:

```python
def test_openai_client_reused_across_embed_calls():
    """OpenAI() must be instantiated exactly once, not once per embed() call."""
    import_calls: list[int] = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            import_calls.append(1)
            self.embeddings = MagicMock()
            self.embeddings.create.return_value = MagicMock(
                data=[MagicMock(embedding=[0.1, 0.2], index=0)]
            )

    with patch("src.internal.document_index.embedding_cache.OpenAI", FakeOpenAI, create=True):
        from src.internal.document_index.embedding_cache import OpenAIEmbedder

        # Force re-import to pick up the mock — if the class caches on __init__
        # import_calls should contain exactly one entry after N embed() calls.
        embedder = OpenAIEmbedder(model="text-embedding-3-small")
        embedder.embed(["hello"])
        embedder.embed(["world"])
        embedder.embed(["again"])

    assert import_calls == [1], (
        f"Expected OpenAI to be instantiated once, got {len(import_calls)} times"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/test_embedding_cache.py::test_openai_client_reused_across_embed_calls -v
```

Expected: FAIL (`import_calls` has 3 entries, one per `embed()` call)

- [ ] **Step 3: Update `OpenAIEmbedder` in `embedding_cache.py`**

In `OpenAIEmbedder.__init__()`, add lazy client cache after the `self._cache` assignment (around line 153):

```python
self._openai_client: Any | None = None  # lazily initialised; avoids ImportError if openai not installed
```

Replace `_call_openai()`:

```python
def _call_openai(self, texts: list[str]) -> np.ndarray:
    if self._openai_client is None:
        from openai import OpenAI
        self._openai_client = OpenAI(api_key=self._api_key)
    rows: list[list[float]] = []
    for start in range(0, len(texts), self._BATCH_SIZE):
        batch = texts[start : start + self._BATCH_SIZE]
        response = self._openai_client.embeddings.create(input=batch, model=self.model)
        rows.extend(
            item.embedding for item in sorted(response.data, key=lambda x: x.index)
        )
    return np.array(rows, dtype=np.float32)
```

The `__init__` type annotation for `_openai_client` uses `Any` so no additional import is needed (the `from __future__ import annotations` is already at the top of the file).

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_embedding_cache.py -v
```

Expected: all tests PASS including the new one

- [ ] **Step 5: Run the full test suite**

```bash
pytest tests/unit/ -v --tb=short -q
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/internal/document_index/embedding_cache.py tests/unit/test_embedding_cache.py
git commit -m "perf: cache OpenAI client instance in OpenAIEmbedder to reuse connection pool"
```

---

## Self-Review

### Spec coverage

| Optimization | Task |
|---|---|
| N individual SQLite commits in `process_batch()` | Task 1 + Task 2 |
| N individual SQLite deletes in `prune_connector()` | Task 3 |
| N×M individual SQLite commits in `sync_doc_permissions()` | Task 3 |
| N OpenSearch round-trips in `id_based_retrieval()` | Task 4 |
| New OpenAI HTTP client on each embed miss | Task 5 |

### Placeholder scan
No TBD/TODO/placeholder steps. All code blocks are complete.

### Type consistency
- `upsert_documents_bulk()` returns `list[StoredDocument]` — matches `upsert_document()` return type.
- `delete_documents_bulk()` takes `list[str]` — matches what `prune_connector()` builds: `[doc.id for doc in stored ...]`.
- `grant_document_access_bulk()` takes `list[DocumentPermission]` — matches what `sync_doc_permissions()` already iterates.
- `msearch()` returns `list[list[SearchHit[DocumentChunkWithoutVectors]]]` — matches the element type that `id_based_retrieval()` already processes via `_convert_retrieved_opensearch_chunk_to_inference_chunk_uncleaned()`.
- `_openai_client: Any | None` — `Any` is imported via `from typing import Any` which is already present in `embedding_cache.py` implicitly through `from __future__ import annotations`.
