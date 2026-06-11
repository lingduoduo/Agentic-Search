"""Tests for DocprocessingWorker bulk upsert path."""

from __future__ import annotations

from unittest.mock import patch
import numpy as np

from src.internal.connectors.models import Document
from src.internal.db.store import AgenticSearchStore
from src.internal.servers.backgroundworker.docprocessing import (
    DocprocessingWorker,
    DocprocessingConfig,
)
from src.internal.document_index import DefaultIndexingEmbedder
from src.internal.document_index.models import ChunkingConfig, EmbeddingConfig


def test_process_batch_uses_bulk_upsert(tmp_path):
    """process_batch must call upsert_documents_bulk, not upsert_document, for each loop."""
    store = AgenticSearchStore(tmp_path / "test.sqlite3")
    worker = DocprocessingWorker(
        store=store,
        chunk_sink=None,
        embedder=DefaultIndexingEmbedder(
            embedding_fn=lambda texts: np.ones((len(texts), 4), dtype=np.float32),
            config=EmbeddingConfig(retrieval_method="contriever"),
        ),
        config=DocprocessingConfig(
            chunking=ChunkingConfig(include_title=False, include_metadata=False)
        ),
    )
    docs = [
        Document(id=f"doc-{i}", title=f"T{i}", contents=f"Content {i}")
        for i in range(5)
    ]

    with (
        patch.object(store, "upsert_document") as mock_single,
        patch.object(
            store, "upsert_documents_bulk", wraps=store.upsert_documents_bulk
        ) as mock_bulk,
    ):
        worker.process_batch(docs)

    mock_single.assert_not_called()
    assert mock_bulk.call_count == 2

    # Verify all docs were persisted to the store
    for i in range(5):
        stored = store.get_document(f"doc-{i}")
        assert stored is not None, f"doc-{i} not persisted"

    # Verify at least one doc has indexed_chunks written back
    indexed = [store.get_document(f"doc-{i}") for i in range(5)]
    assert any("indexed_chunks" in (d.metadata or {}) for d in indexed), (
        "No doc had indexed_chunks written back to the store"
    )

    store.close()
