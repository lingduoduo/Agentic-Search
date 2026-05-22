"""Unit tests for the lightweight indexing pipeline."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.connectors import Document
from src.retrieval.index_builder import (
    ChunkingConfig,
    EmbeddingConfig,
    IndexingPipelineConfig,
    IndexWriterConfig,
    chunk_document,
    deterministic_embedding_fn,
    embed_chunks,
    run_indexing_pipeline,
    write_faiss_index,
)


def test_chunk_document_splits_with_overlap_and_title():
    document = Document(
        id="doc",
        title="Title",
        contents="abcdefghijklmnopqrstuvwxyz",
        metadata={"source": "unit"},
        permissions={"public": True},
    )

    chunks = chunk_document(
        document,
        ChunkingConfig(chunk_size=10, chunk_overlap=2),
    )

    assert [chunk.chunk_id for chunk in chunks] == [0, 1, 2]
    assert chunks[0].id == "doc::chunk-0"
    assert chunks[0].text == "Title\nabcdefghij"
    assert chunks[1].text == "Title\nijklmnopqr"
    assert chunks[0].metadata == {
        "source": "unit",
        "permissions": {"public": True},
    }


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
