"""Tests for DocprocessingWorker bulk upsert path."""

from __future__ import annotations

from unittest.mock import patch
import numpy as np

from src.backend.connectors.models import Document
from src.backend.db.store import AgenticSearchStore
from src.backend.servers.backgroundworker.docprocessing import (
    DocprocessingWorker,
    DocprocessingConfig,
)
from src.backend.document_index import DefaultIndexingEmbedder
from src.backend.document_index.models import ChunkingConfig, EmbeddingConfig


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
    assert mock_bulk.call_count >= 1
    store.close()
