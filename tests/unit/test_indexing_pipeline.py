"""Unit tests for the lightweight indexing pipeline."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.connectors import Document
from src.retrieval.index_builder import (
    ChunkingConfig,
    EmbeddingConfig,
    IndexChunk,
    IndexingPipelineConfig,
    IndexWriterConfig,
    chunk_document,
    deterministic_embedding_fn,
    embed_chunks,
    embed_chunks_with_failure_handling,
    run_indexing_pipeline,
    write_faiss_index,
)
from src.retrieval.indexing_heartbeat import IndexingHeartbeatInterface


class RecordingHeartbeat(IndexingHeartbeatInterface):
    def __init__(self, stop_after_checks: int | None = None) -> None:
        self.stop_after_checks = stop_after_checks
        self.checks = 0
        self.events: list[tuple[str, int]] = []

    def should_stop(self) -> bool:
        self.checks += 1
        return (
            self.stop_after_checks is not None and self.checks >= self.stop_after_checks
        )

    def progress(self, tag: str, amount: int) -> None:
        self.events.append((tag, amount))


def test_chunk_document_splits_with_overlap_and_title():
    document = Document(
        id="doc",
        title="Title",
        contents=("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"),
        metadata={"source": "unit"},
        permissions={"public": True},
    )

    chunks = chunk_document(
        document,
        ChunkingConfig(
            chunk_size=10,
            chunk_overlap=2,
            min_content_tokens=2,
            max_metadata_percentage=1.0,
        ),
    )

    assert [chunk.chunk_id for chunk in chunks] == [0, 1, 2, 3]
    assert chunks[0].id == "doc::chunk-0"
    assert chunks[0].text == (
        "Title\n\nalpha beta gamma delta epsilon\n\nMetadata:\n\tsource - unit"
    )
    assert chunks[1].text == (
        "Title\n\ndelta epsilon zeta eta theta\n\nMetadata:\n\tsource - unit"
    )
    assert chunks[0].metadata == {
        "source": "unit",
        "permissions": {"public": True},
        "metadata_keyword": "\n\nunit",
    }
    assert chunks[0].blurb == "alpha beta gamma delta epsilon"


def test_chunk_document_drops_metadata_when_budget_is_tight():
    document = Document(
        id="doc",
        title="Important title",
        contents="one two three four five six seven eight",
        metadata={"tags": ["alpha", "beta", "gamma", "delta"]},
    )

    chunks = chunk_document(
        document,
        ChunkingConfig(chunk_size=8, chunk_overlap=0, min_content_tokens=2),
    )

    assert chunks[0].text.startswith("Important title\n\n")
    assert "Metadata:" not in chunks[0].text
    assert chunks[0].metadata["metadata_keyword"] == "\n\nalpha beta gamma delta"


def test_chunk_document_can_emit_large_chunks():
    document = Document(
        id="doc",
        contents="one two three four five six seven eight nine ten eleven twelve",
    )

    chunks = chunk_document(
        document,
        ChunkingConfig(
            chunk_size=4,
            chunk_overlap=0,
            include_title=False,
            include_metadata=False,
            enable_large_chunks=True,
            large_chunk_ratio=2,
        ),
    )

    assert [chunk.id for chunk in chunks] == [
        "doc::chunk-0",
        "doc::chunk-1",
        "doc::chunk-2",
        "doc::large-chunk-0",
    ]
    assert chunks[-1].large_chunk_reference_ids == [0, 1]
    assert "\n\n---\n\n" in chunks[-1].text


def test_embed_chunks_uses_index_builder_text_preparation():
    chunks = chunk_document(
        Document(id="doc", title=None, contents="alpha beta"),
        ChunkingConfig(chunk_size=50, chunk_overlap=0, include_title=False),
    )
    seen_texts = []

    def fake_embed(texts):
        seen_texts.extend(texts)
        return np.ones((len(texts), 3), dtype=np.float32)

    embedded = embed_chunks(
        chunks,
        embedding_fn=fake_embed,
        config=EmbeddingConfig(retrieval_method="e5", batch_size=1),
    )

    assert seen_texts == ["passage: alpha beta"]
    assert embedded[0].embedding.tolist() == [1.0, 1.0, 1.0]


def test_embed_chunks_can_cache_title_embeddings_and_normalize():
    chunks = chunk_document(
        Document(id="doc", title="Shared", contents="alpha beta gamma delta"),
        ChunkingConfig(chunk_size=2, chunk_overlap=0),
    )
    seen_texts = []

    def fake_embed(texts):
        seen_texts.extend(texts)
        return np.array([[3.0, 4.0] for _ in texts], dtype=np.float32)

    embedded = embed_chunks(
        chunks,
        embedding_fn=fake_embed,
        config=EmbeddingConfig(
            retrieval_method="contriever",
            batch_size=1,
            embed_titles=True,
            normalize_embeddings=True,
            passage_prefix="doc:",
        ),
    )

    assert seen_texts.count("doc: Shared") == 1
    np.testing.assert_allclose(embedded[0].embedding, [0.6, 0.8])
    assert embedded[1].title_embedding is not None
    np.testing.assert_allclose(embedded[0].title_embedding, [0.6, 0.8])


def test_embed_chunks_with_failure_handling_isolates_bad_document():
    chunks = [
        IndexChunk(id="good::chunk-0", document_id="good", chunk_id=0, text="good"),
        IndexChunk(id="bad::chunk-0", document_id="bad", chunk_id=0, text="bad"),
    ]
    calls = []

    def fake_embed(texts):
        calls.append(list(texts))
        if any("bad" in text for text in texts):
            raise RuntimeError("bad embedding")
        return np.ones((len(texts), 2), dtype=np.float32)

    embedded, failures = embed_chunks_with_failure_handling(
        chunks,
        embedding_fn=fake_embed,
        config=EmbeddingConfig(retrieval_method="contriever", batch_size=8),
    )

    assert [item.chunk.document_id for item in embedded] == ["good"]
    assert failures[0].document_id == "bad"
    assert failures[0].exception_type == "RuntimeError"
    assert calls == [["good", "bad"], ["good"], ["bad"]]


def test_embed_chunks_rejects_mismatched_embedding_rows():
    chunks = chunk_document(
        Document(id="doc", contents="alpha beta"),
        ChunkingConfig(chunk_size=50, chunk_overlap=0),
    )

    with pytest.raises(ValueError, match="one row per input"):
        embed_chunks(
            chunks,
            embedding_fn=lambda texts: np.ones((len(texts) + 1, 2)),
            config=EmbeddingConfig(),
        )


def test_run_indexing_pipeline_writes_corpus_and_embeddings(tmp_path):
    documents = [
        Document(id="one", title="One", contents="alpha beta"),
        Document(id="two", title="Two", contents="gamma delta"),
    ]
    config = IndexingPipelineConfig(
        chunking=ChunkingConfig(chunk_size=100, chunk_overlap=0),
        embedding=EmbeddingConfig(retrieval_method="contriever", batch_size=2),
        writer=IndexWriterConfig(save_dir=tmp_path),
    )

    result = run_indexing_pipeline(
        documents,
        config=config,
        embedding_fn=deterministic_embedding_fn(dim=4),
    )

    assert result.total_documents == 2
    assert result.total_chunks == 2
    rows = [
        json.loads(line)
        for line in result.corpus_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in rows] == ["one::chunk-0", "two::chunk-0"]
    embeddings = np.memmap(result.embedding_path, mode="r", dtype=np.float32).reshape(
        2, 4
    )
    assert embeddings.shape == (2, 4)
    assert result.index_path is None


def test_run_indexing_pipeline_reports_heartbeat_progress(tmp_path):
    documents = [
        Document(id="one", title="One", contents="alpha beta"),
        Document(id="two", title="Two", contents="gamma delta"),
    ]
    config = IndexingPipelineConfig(
        chunking=ChunkingConfig(chunk_size=100, chunk_overlap=0),
        embedding=EmbeddingConfig(retrieval_method="contriever", batch_size=1),
        writer=IndexWriterConfig(save_dir=tmp_path),
    )
    heartbeat = RecordingHeartbeat()

    run_indexing_pipeline(
        documents,
        config=config,
        embedding_fn=deterministic_embedding_fn(dim=4),
        callback=heartbeat,
    )

    assert ("chunk_documents", 1) in heartbeat.events
    assert heartbeat.events.count(("embed_chunks", 1)) == 2
    assert ("write_corpus_jsonl", 2) in heartbeat.events
    assert ("write_embeddings_memmap", 2) in heartbeat.events


def test_run_indexing_pipeline_honors_heartbeat_stop(tmp_path):
    config = IndexingPipelineConfig(
        chunking=ChunkingConfig(chunk_size=100, chunk_overlap=0),
        embedding=EmbeddingConfig(retrieval_method="contriever", batch_size=1),
        writer=IndexWriterConfig(save_dir=tmp_path),
    )
    heartbeat = RecordingHeartbeat(stop_after_checks=1)

    with pytest.raises(RuntimeError, match="stop signal detected"):
        run_indexing_pipeline(
            [Document(id="one", title="One", contents="alpha beta")],
            config=config,
            embedding_fn=deterministic_embedding_fn(dim=4),
            callback=heartbeat,
        )


def test_write_faiss_index_delegates_to_index_builder(monkeypatch, tmp_path):
    chunks = chunk_document(
        Document(id="doc", contents="alpha beta"),
        ChunkingConfig(chunk_size=50, chunk_overlap=0),
    )
    embedded = embed_chunks(
        chunks,
        embedding_fn=deterministic_embedding_fn(dim=2),
        config=EmbeddingConfig(),
    )
    calls = []

    def fake_write_dense_faiss_index(embeddings, index_path, **kwargs):
        calls.append((embeddings.copy(), index_path, kwargs))
        index_path.write_text("index", encoding="utf-8")

    monkeypatch.setattr(
        "src.retrieval.index_builder.write_dense_faiss_index",
        fake_write_dense_faiss_index,
    )

    path = write_faiss_index(
        embedded,
        tmp_path / "dense.index",
        faiss_type="HNSW64",
        hnsw_ef_construction=64,
        hnsw_ef_search=32,
    )

    assert path.read_text(encoding="utf-8") == "index"
    assert calls[0][0].shape == (1, 2)
    assert calls[0][1] == path
    assert calls[0][2] == {
        "faiss_type": "HNSW64",
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 32,
    }
