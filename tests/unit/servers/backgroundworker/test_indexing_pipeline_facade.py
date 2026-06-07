"""Tests for the consolidated document-indexing facade."""

from __future__ import annotations

import numpy as np

from src.backend.connectors import Document
from src.backend.document_index import ChunkBatchStore
from src.backend.document_index import Chunker
from src.backend.document_index import DefaultIndexingEmbedder
from src.backend.document_index import embed_and_stream
from src.backend.document_index import filter_documents
from src.backend.document_index import index_document_batch
from src.backend.document_index import index_documents
from src.backend.document_index import write_chunks_with_backoff
from src.backend.document_index.index_builder import IndexingHeartbeatInterface
from src.backend.document_index.models import ChunkingConfig
from src.backend.document_index.models import EmbeddingConfig


def test_chunker_and_embedder_facade_support_mini_chunks():
    chunker = Chunker(
        ChunkingConfig(
            chunk_size=5,
            chunk_overlap=0,
            include_title=False,
            include_metadata=False,
            enable_mini_chunks=True,
            mini_chunk_size=2,
        )
    )
    chunks = chunker.chunk([Document(id="doc", contents="one two three four five")])

    def fake_embed(texts):
        return np.ones((len(texts), 3), dtype=np.float32)

    embedded = DefaultIndexingEmbedder(
        embedding_fn=fake_embed,
        config=EmbeddingConfig(retrieval_method="contriever"),
    ).embed_chunks(chunks)

    assert chunks[0].mini_chunk_texts == ["one two", "three four", "five"]
    assert embedded[0].embedding.shape == (3,)
    assert len(embedded[0].mini_chunk_embeddings) == 3


def test_chunk_batch_store_streams_and_scrubs_failed_docs():
    chunker = Chunker(
        ChunkingConfig(chunk_size=10, chunk_overlap=0, include_title=False)
    )
    chunks = chunker.chunk(
        [
            Document(id="ok", contents="alpha"),
            Document(id="bad", contents="beta"),
        ]
    )
    embedder = DefaultIndexingEmbedder(
        embedding_fn=lambda texts: np.ones((len(texts), 2), dtype=np.float32),
        config=EmbeddingConfig(retrieval_method="contriever"),
    )
    embedded = embedder.embed_chunks(chunks)

    with ChunkBatchStore() as store:
        store.save(embedded, batch_idx=2)
        assert [chunk.chunk.document_id for chunk in store.stream()] == ["ok", "bad"]

        store.scrub_failed_docs({"bad"})

        assert [chunk.chunk.document_id for chunk in store.stream()] == ["ok"]


def test_filter_documents_and_index_document_batch(tmp_path):
    kept, failures = filter_documents(
        [
            Document(id="empty", contents=" "),
            Document(id="ok", contents="alpha"),
        ]
    )

    assert [document.id for document in kept] == ["ok"]
    assert [failure.document_id for failure in failures] == ["empty"]

    result = index_document_batch(
        kept,
        save_dir=str(tmp_path),
        chunking=ChunkingConfig(chunk_size=10, chunk_overlap=0),
        embedding=EmbeddingConfig(retrieval_method="contriever"),
        embedding_fn=lambda texts: np.ones((len(texts), 2), dtype=np.float32),
    )

    assert result.total_documents == 1
    assert result.total_chunks == 1
    assert result.corpus_path.exists()


def test_embed_and_stream_yields_store_and_prefilter_failures():
    with embed_and_stream(
        [
            Document(id="empty", contents=" "),
            Document(id="ok", contents="alpha"),
            Document(id="ok", contents="duplicate"),
        ],
        embedder=DefaultIndexingEmbedder(
            embedding_fn=lambda texts: np.ones((len(texts), 2), dtype=np.float32),
            config=EmbeddingConfig(retrieval_method="contriever"),
        ),
    ) as (failures, store):
        assert [failure.document_id for failure in failures] == ["empty", "ok"]
        assert failures[1].message == "Duplicate document id skipped before indexing."
        assert [chunk.chunk.document_id for chunk in store.stream()] == ["ok"]


def test_embed_and_stream_persists_embedding_batches_incrementally():
    batch_sizes: list[int] = []

    def fake_embed(texts):
        batch_sizes.append(len(texts))
        return np.ones((len(texts), 2), dtype=np.float32)

    with embed_and_stream(
        [
            Document(id="one", contents="alpha"),
            Document(id="two", contents="beta"),
            Document(id="three", contents="gamma"),
        ],
        chunking=ChunkingConfig(chunk_size=10, chunk_overlap=0, include_title=False),
        embedder=DefaultIndexingEmbedder(
            embedding_fn=fake_embed,
            config=EmbeddingConfig(retrieval_method="contriever", batch_size=2),
        ),
    ) as (failures, store):
        assert failures == []
        assert [chunk.chunk.document_id for chunk in store.stream()] == [
            "one",
            "two",
            "three",
        ]

    assert batch_sizes == [2, 1]


def test_write_chunks_with_backoff_isolates_document_failures():
    chunker = Chunker(
        ChunkingConfig(chunk_size=10, chunk_overlap=0, include_title=False)
    )
    embedded = DefaultIndexingEmbedder(
        embedding_fn=lambda texts: np.ones((len(texts), 2), dtype=np.float32),
        config=EmbeddingConfig(retrieval_method="contriever"),
    ).embed_chunks(
        chunker.chunk(
            [
                Document(id="ok", contents="alpha"),
                Document(id="bad", contents="beta"),
            ]
        )
    )

    class Sink:
        def __init__(self):
            self.calls = 0

        def index(self, chunks):
            self.calls += 1
            rows = list(chunks)
            if self.calls == 1 or rows[0].chunk.document_id == "bad":
                raise RuntimeError("write failed")
            return [row.chunk.id for row in rows]

    records, failures = write_chunks_with_backoff(Sink(), lambda: iter(embedded))

    assert records == ["ok::chunk-0"]
    assert [failure.document_id for failure in failures] == ["bad"]


def test_index_documents_runs_the_complete_pipeline():
    indexed_document_ids: list[str] = []

    class Sink:
        def index(self, chunks):
            rows = list(chunks)
            indexed_document_ids.extend(row.chunk.document_id for row in rows)
            return [row.chunk.id for row in rows]

    result = index_documents(
        [
            Document(id="empty", contents=" "),
            Document(id="ok", contents="alpha"),
        ],
        sink=Sink(),
        chunking=ChunkingConfig(chunk_size=10, chunk_overlap=0, include_title=False),
        embedder=DefaultIndexingEmbedder(
            embedding_fn=lambda texts: np.ones((len(texts), 2), dtype=np.float32),
            config=EmbeddingConfig(retrieval_method="contriever"),
        ),
    )

    assert indexed_document_ids == ["ok"]
    assert result.successful_chunk_counts == {"ok": 1}
    assert [failure.document_id for failure in result.failures] == ["empty"]


def test_index_documents_does_not_report_stopped_chunks_as_successful():
    class StopAfterChunking(IndexingHeartbeatInterface):
        def __init__(self):
            self.checks = 0

        def should_stop(self):
            self.checks += 1
            return self.checks >= 3

        def progress(self, tag, amount):
            pass

    result = index_documents(
        [Document(id="doc", contents="alpha")],
        callback=StopAfterChunking(),
        chunking=ChunkingConfig(chunk_size=10, chunk_overlap=0, include_title=False),
    )

    assert result.stopped is True
    assert result.successful_chunk_counts == {}
