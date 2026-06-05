"""Unit tests for the lightweight indexing pipeline."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.backend.connectors import Document
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
    filter_indexable_documents,
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


def test_chunk_document_can_emit_mini_chunks_for_multipass_indexing():
    document = Document(
        id="doc",
        contents="one two three four five six seven eight",
    )

    chunks = chunk_document(
        document,
        ChunkingConfig(
            chunk_size=8,
            chunk_overlap=0,
            include_title=False,
            include_metadata=False,
            enable_mini_chunks=True,
            mini_chunk_size=3,
        ),
    )

    assert chunks[0].mini_chunk_texts == [
        "one two three",
        "four five six",
        "seven eight",
    ]


def test_filter_indexable_documents_skips_duplicate_document_ids():
    kept, failures = filter_indexable_documents(
        [
            Document(id="doc", title="First", contents="alpha", url="https://one"),
            Document(id="doc", title="Second", contents="beta", url="https://two"),
            Document(id="other", title="Other", contents="gamma"),
        ]
    )

    assert [document.title for document in kept] == ["First", "Other"]
    assert [failure.document_id for failure in failures] == ["doc"]
    assert failures[0].message == "Duplicate document id skipped before indexing."
    assert failures[0].metadata == {"url": "https://two"}


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


def test_embed_chunks_maps_mini_chunk_embeddings():
    chunks = chunk_document(
        Document(id="doc", title=None, contents="alpha beta gamma delta epsilon"),
        ChunkingConfig(
            chunk_size=5,
            chunk_overlap=0,
            include_title=False,
            include_metadata=False,
            enable_mini_chunks=True,
            mini_chunk_size=2,
        ),
    )
    seen_texts = []

    def fake_embed(texts):
        seen_texts.extend(texts)
        return np.array(
            [[float(index), float(index + 10)] for index, _ in enumerate(texts)],
            dtype=np.float32,
        )

    embedded = embed_chunks(
        chunks,
        embedding_fn=fake_embed,
        config=EmbeddingConfig(retrieval_method="contriever", batch_size=1),
    )

    assert seen_texts == [
        "alpha beta gamma delta epsilon",
        "alpha beta",
        "gamma delta",
        "epsilon",
    ]
    np.testing.assert_allclose(embedded[0].embedding, [0.0, 10.0])
    assert len(embedded[0].mini_chunk_embeddings) == 3
    np.testing.assert_allclose(embedded[0].mini_chunk_embeddings[1], [2.0, 12.0])


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


def test_filter_indexable_documents_reports_empty_and_oversized_docs():
    kept, failures = filter_indexable_documents(
        [
            Document(id="empty", title=" ", contents=" "),
            Document(id="large", title="Title", contents="x" * 20, url="https://l"),
            Document(id="ok", title="OK", contents="small"),
        ],
        max_document_chars=10,
    )

    assert [document.id for document in kept] == ["ok"]
    assert [failure.document_id for failure in failures] == ["empty", "large"]
    assert failures[0].message == "Document has neither title nor contents."
    assert failures[1].metadata["char_count"] == 25


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
    assert rows[0]["mini_chunk_texts"] == []
    embeddings = np.memmap(result.embedding_path, mode="r", dtype=np.float32).reshape(
        2, 4
    )
    assert embeddings.shape == (2, 4)
    assert result.index_path is None


def test_run_indexing_pipeline_returns_prefilter_failures(tmp_path):
    documents = [
        Document(id="ok", title="OK", contents="alpha"),
        Document(id="large", title=None, contents="x" * 20),
    ]
    config = IndexingPipelineConfig(
        chunking=ChunkingConfig(
            chunk_size=100,
            chunk_overlap=0,
            max_document_chars=10,
        ),
        embedding=EmbeddingConfig(retrieval_method="contriever", batch_size=2),
        writer=IndexWriterConfig(save_dir=tmp_path),
    )

    result = run_indexing_pipeline(
        documents,
        config=config,
        embedding_fn=deterministic_embedding_fn(dim=4),
    )

    assert result.total_documents == 2
    assert result.total_chunks == 1
    assert [failure.document_id for failure in result.failures] == ["large"]


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


# ---------------------------------------------------------------------------
# Paragraph-aware chunking
# ---------------------------------------------------------------------------

from src.retrieval.index_builder import _split_paragraphs, _split_sentences_in_paragraph  # noqa: E402


def test_split_paragraphs_splits_on_double_newline():
    text = "First paragraph with sentences.\n\nSecond paragraph here."
    result = _split_paragraphs(text)
    assert len(result) == 2
    assert result[0] == "First paragraph with sentences."
    assert result[1] == "Second paragraph here."


def test_split_paragraphs_splits_on_markdown_header():
    text = "Intro text here.\n## Section Two\nSection two content."
    result = _split_paragraphs(text)
    assert len(result) == 2
    assert "Intro text" in result[0]
    assert "Section Two" in result[1] or "Section two" in result[1]


def test_split_paragraphs_single_paragraph_returns_one_item():
    text = "Just one sentence. And another. And a third."
    result = _split_paragraphs(text)
    assert len(result) == 1


def test_split_paragraphs_empty_returns_empty():
    assert _split_paragraphs("") == []
    assert _split_paragraphs("   \n\n  ") == []


def test_split_sentences_in_paragraph_splits_on_punctuation():
    para = "First sentence. Second sentence! Third sentence?"
    result = _split_sentences_in_paragraph(para)
    assert len(result) == 3
    assert result[0] == "First sentence."


def test_split_sentences_in_paragraph_does_not_collapse_paragraphs():
    para = "One sentence. Two sentence."
    result = _split_sentences_in_paragraph(para)
    assert len(result) == 2


def test_chunk_document_does_not_span_section_boundary():
    """Chunks should respect paragraph boundaries — no chunk should span two unrelated sections."""
    section_a = " ".join(["word"] * 60)
    section_b = " ".join(["term"] * 60)
    document = Document(
        id="doc-sections",
        title="Test",
        contents=f"{section_a}\n\n{section_b}",
        metadata={},
        permissions={},
    )
    chunks = chunk_document(
        document,
        ChunkingConfig(
            chunk_size=50, chunk_overlap=5, include_title=False, include_metadata=False
        ),
    )
    for chunk in chunks:
        word_count = chunk.text.count("word")
        term_count = chunk.text.count("term")
        if word_count > 0 and term_count > 0:
            assert min(word_count, term_count) <= 5, (
                f"Chunk spans sections: {word_count} 'word' tokens and {term_count} 'term' tokens"
            )


def test_chunk_document_sets_section_continuation_on_non_first_chunks():
    """section_continuation should be True for every chunk after the first."""
    document = Document(
        id="doc-cont",
        title="Title",
        contents=" ".join(["sentence."] * 30),
        metadata={},
        permissions={},
    )
    chunks = chunk_document(
        document,
        ChunkingConfig(
            chunk_size=10, chunk_overlap=2, include_title=False, include_metadata=False
        ),
    )
    assert len(chunks) >= 2, "Need at least 2 chunks to test continuation flag"
    assert chunks[0].section_continuation is False
    for chunk in chunks[1:]:
        assert chunk.section_continuation is True, (
            f"chunk_id={chunk.chunk_id} should have section_continuation=True"
        )
