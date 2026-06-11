"""Unit tests for the streaming indexing pipeline facade."""

from __future__ import annotations


from src.internal.connectors.models import Document
from src.internal.document_index.index_builder import deterministic_embedding_fn
from src.internal.servers.indexing import (
    BatchIndexingResult,
    ChunkBatchStore,
    StreamingIndexingConfig,
    StreamingPipelineResult,
    run_indexing_pipeline,
    stream_indexing_pipeline,
)


def _make_doc(doc_id: str, contents: str = "hello world foo bar baz") -> Document:
    return Document(
        id=doc_id,
        title=f"Title {doc_id}",
        contents=contents,
        metadata={},
        permissions={"public": True},
    )


def _tiny_cfg() -> StreamingIndexingConfig:
    from src.internal.document_index.models import ChunkingConfig, EmbeddingConfig

    return StreamingIndexingConfig(
        chunking=ChunkingConfig(chunk_size=8, chunk_overlap=2, min_content_tokens=1),
        embedding=EmbeddingConfig(batch_size=4),
        docs_per_batch=2,
    )


# ---------------------------------------------------------------------------
# ChunkBatchStore tests
# ---------------------------------------------------------------------------


def test_chunk_batch_store_save_and_stream():
    from src.internal.document_index.models import ChunkingConfig, EmbeddingConfig

    cfg = StreamingIndexingConfig(
        chunking=ChunkingConfig(chunk_size=8, chunk_overlap=0, min_content_tokens=1),
        embedding=EmbeddingConfig(batch_size=4),
        docs_per_batch=10,
    )
    embed_fn = deterministic_embedding_fn(8)
    docs = [_make_doc(f"d{i}") for i in range(3)]

    with ChunkBatchStore() as store:
        run_indexing_pipeline(docs, embedding_fn=embed_fn, store=store, config=cfg)
        all_chunks = list(store.stream())

    assert len(all_chunks) > 0
    assert all(ec.embedding.shape[-1] == 8 for ec in all_chunks)


def test_chunk_batch_store_scrub_removes_failed_docs():
    embed_fn = deterministic_embedding_fn(8)
    cfg = _tiny_cfg()
    docs = [_make_doc("keep1"), _make_doc("keep2"), _make_doc("scrub_me")]

    with ChunkBatchStore() as store:
        run_indexing_pipeline(docs, embedding_fn=embed_fn, store=store, config=cfg)
        store.scrub_failed_docs({"scrub_me"})
        remaining = list(store.stream())

    doc_ids = {ec.chunk.document_id for ec in remaining}
    assert "scrub_me" not in doc_ids
    assert "keep1" in doc_ids or "keep2" in doc_ids


def test_chunk_batch_store_stream_multiple_times():
    embed_fn = deterministic_embedding_fn(8)
    cfg = _tiny_cfg()
    docs = [_make_doc("d1"), _make_doc("d2")]

    with ChunkBatchStore() as store:
        run_indexing_pipeline(docs, embedding_fn=embed_fn, store=store, config=cfg)
        first = list(store.stream())
        second = list(store.stream())

    assert len(first) == len(second)


# ---------------------------------------------------------------------------
# stream_indexing_pipeline tests
# ---------------------------------------------------------------------------


def test_stream_pipeline_yields_one_result_per_batch():
    embed_fn = deterministic_embedding_fn(8)
    cfg = _tiny_cfg()  # docs_per_batch=2
    docs = [_make_doc(f"d{i}") for i in range(5)]

    with ChunkBatchStore() as store:
        results = list(
            stream_indexing_pipeline(
                docs, embedding_fn=embed_fn, store=store, config=cfg
            )
        )

    # 5 docs / 2 per batch = ceil(5/2) = 3 batches
    assert len(results) == 3
    assert all(isinstance(r, BatchIndexingResult) for r in results)


def test_stream_pipeline_batch_indices_are_sequential():
    embed_fn = deterministic_embedding_fn(8)
    cfg = _tiny_cfg()
    docs = [_make_doc(f"d{i}") for i in range(4)]

    with ChunkBatchStore() as store:
        results = list(
            stream_indexing_pipeline(
                docs, embedding_fn=embed_fn, store=store, config=cfg
            )
        )

    assert [r.batch_idx for r in results] == [0, 1]


def test_stream_pipeline_counts_correct():
    embed_fn = deterministic_embedding_fn(8)
    from src.internal.document_index.models import ChunkingConfig, EmbeddingConfig

    cfg = StreamingIndexingConfig(
        chunking=ChunkingConfig(chunk_size=8, chunk_overlap=0, min_content_tokens=1),
        embedding=EmbeddingConfig(batch_size=4),
        docs_per_batch=10,
    )
    docs = [_make_doc(f"d{i}") for i in range(3)]

    with ChunkBatchStore() as store:
        results = list(
            stream_indexing_pipeline(
                docs, embedding_fn=embed_fn, store=store, config=cfg
            )
        )

    assert len(results) == 1
    r = results[0]
    assert r.docs_in_batch == 3
    assert r.chunks_produced > 0
    assert r.chunks_embedded == r.chunks_produced
    assert r.failures == []


def test_stream_pipeline_prefilter_empty_doc():
    embed_fn = deterministic_embedding_fn(8)
    from src.internal.document_index.models import ChunkingConfig, EmbeddingConfig

    cfg = StreamingIndexingConfig(
        chunking=ChunkingConfig(chunk_size=8, chunk_overlap=0, min_content_tokens=1),
        embedding=EmbeddingConfig(batch_size=4),
        docs_per_batch=10,
    )
    docs = [
        _make_doc("good"),
        Document(id="empty", title="", contents="", metadata={}, permissions={}),
    ]

    with ChunkBatchStore() as store:
        results = list(
            stream_indexing_pipeline(
                docs, embedding_fn=embed_fn, store=store, config=cfg
            )
        )

    assert len(results) == 1
    r = results[0]
    assert len(r.failures) == 1
    assert r.failures[0].document_id == "empty"
    assert r.chunks_embedded >= 1


def test_stream_pipeline_prefilter_duplicate_doc():
    embed_fn = deterministic_embedding_fn(8)
    from src.internal.document_index.models import ChunkingConfig, EmbeddingConfig

    cfg = StreamingIndexingConfig(
        chunking=ChunkingConfig(chunk_size=8, chunk_overlap=0, min_content_tokens=1),
        embedding=EmbeddingConfig(batch_size=4),
        docs_per_batch=10,
    )
    doc = _make_doc("dup")
    docs = [doc, doc]

    with ChunkBatchStore() as store:
        results = list(
            stream_indexing_pipeline(
                docs, embedding_fn=embed_fn, store=store, config=cfg
            )
        )

    assert len(results) == 1
    r = results[0]
    assert len(r.failures) == 1
    assert r.failures[0].document_id == "dup"


def test_stream_pipeline_all_prefilter_failures_yields_result():
    embed_fn = deterministic_embedding_fn(8)
    from src.internal.document_index.models import ChunkingConfig, EmbeddingConfig

    cfg = StreamingIndexingConfig(
        chunking=ChunkingConfig(chunk_size=8, chunk_overlap=0, min_content_tokens=1),
        embedding=EmbeddingConfig(batch_size=4),
        docs_per_batch=10,
    )
    docs = [
        Document(id="e1", title="", contents="", metadata={}, permissions={}),
        Document(id="e2", title="", contents="", metadata={}, permissions={}),
    ]

    with ChunkBatchStore() as store:
        results = list(
            stream_indexing_pipeline(
                docs, embedding_fn=embed_fn, store=store, config=cfg
            )
        )

    assert len(results) == 1
    r = results[0]
    assert r.chunks_produced == 0
    assert r.chunks_embedded == 0
    assert len(r.failures) == 2


def test_stream_pipeline_stop_signal_halts_iteration():
    class StopAfterFirst:
        def __init__(self) -> None:
            self._calls = 0

        def should_stop(self) -> bool:
            self._calls += 1
            return self._calls > 1

        def progress(self, tag: str, amount: int) -> None:
            pass

    embed_fn = deterministic_embedding_fn(8)
    cfg = _tiny_cfg()  # docs_per_batch=2
    docs = [_make_doc(f"d{i}") for i in range(6)]

    with ChunkBatchStore() as store:
        results = list(
            stream_indexing_pipeline(
                docs,
                embedding_fn=embed_fn,
                store=store,
                config=cfg,
                callback=StopAfterFirst(),
            )
        )

    assert len(results) < 3


def test_stream_pipeline_progress_reported():
    class TrackingCallback:
        def __init__(self) -> None:
            self.events: list[tuple[str, int]] = []

        def should_stop(self) -> bool:
            return False

        def progress(self, tag: str, amount: int) -> None:
            self.events.append((tag, amount))

    embed_fn = deterministic_embedding_fn(8)
    from src.internal.document_index.models import ChunkingConfig, EmbeddingConfig

    cfg = StreamingIndexingConfig(
        chunking=ChunkingConfig(chunk_size=8, chunk_overlap=0, min_content_tokens=1),
        embedding=EmbeddingConfig(batch_size=4),
        docs_per_batch=10,
    )
    docs = [_make_doc(f"d{i}") for i in range(2)]
    cb = TrackingCallback()

    with ChunkBatchStore() as store:
        list(
            stream_indexing_pipeline(
                docs, embedding_fn=embed_fn, store=store, config=cfg, callback=cb
            )
        )

    tags = [e[0] for e in cb.events]
    assert "batch_embedded" in tags


# ---------------------------------------------------------------------------
# run_indexing_pipeline tests
# ---------------------------------------------------------------------------


def test_run_pipeline_returns_aggregate():
    embed_fn = deterministic_embedding_fn(8)
    cfg = _tiny_cfg()
    docs = [_make_doc(f"d{i}") for i in range(4)]

    with ChunkBatchStore() as store:
        result = run_indexing_pipeline(
            docs, embedding_fn=embed_fn, store=store, config=cfg
        )

    assert isinstance(result, StreamingPipelineResult)
    assert result.total_documents == 4
    assert result.total_chunks > 0
    assert result.total_embedded == result.total_chunks
    assert result.failures == []


def test_run_pipeline_reports_prefilter_failures():
    embed_fn = deterministic_embedding_fn(8)
    from src.internal.document_index.models import ChunkingConfig, EmbeddingConfig

    cfg = StreamingIndexingConfig(
        chunking=ChunkingConfig(chunk_size=8, chunk_overlap=0, min_content_tokens=1),
        embedding=EmbeddingConfig(batch_size=4),
        docs_per_batch=10,
    )
    docs = [
        _make_doc("good"),
        Document(id="bad", title="", contents="", metadata={}, permissions={}),
    ]

    with ChunkBatchStore() as store:
        result = run_indexing_pipeline(
            docs, embedding_fn=embed_fn, store=store, config=cfg
        )

    assert result.total_documents == 2
    assert len(result.failures) == 1
    assert result.failures[0].document_id == "bad"


def test_run_pipeline_empty_documents():
    embed_fn = deterministic_embedding_fn(8)
    cfg = _tiny_cfg()

    with ChunkBatchStore() as store:
        result = run_indexing_pipeline(
            [], embedding_fn=embed_fn, store=store, config=cfg
        )

    assert result.total_documents == 0
    assert result.total_chunks == 0
    assert result.total_embedded == 0
    assert result.failures == []


def test_run_pipeline_store_has_all_embedded_chunks():
    embed_fn = deterministic_embedding_fn(8)
    from src.internal.document_index.models import ChunkingConfig, EmbeddingConfig

    cfg = StreamingIndexingConfig(
        chunking=ChunkingConfig(chunk_size=8, chunk_overlap=0, min_content_tokens=1),
        embedding=EmbeddingConfig(batch_size=4),
        docs_per_batch=10,
    )
    docs = [_make_doc(f"d{i}") for i in range(3)]

    with ChunkBatchStore() as store:
        result = run_indexing_pipeline(
            docs, embedding_fn=embed_fn, store=store, config=cfg
        )
        chunks_from_store = list(store.stream())

    assert len(chunks_from_store) == result.total_embedded


def test_run_pipeline_embedded_chunk_shape():
    dim = 16
    embed_fn = deterministic_embedding_fn(dim)
    from src.internal.document_index.models import ChunkingConfig, EmbeddingConfig

    cfg = StreamingIndexingConfig(
        chunking=ChunkingConfig(chunk_size=8, chunk_overlap=0, min_content_tokens=1),
        embedding=EmbeddingConfig(batch_size=4),
        docs_per_batch=10,
    )
    docs = [_make_doc("d1")]

    with ChunkBatchStore() as store:
        run_indexing_pipeline(docs, embedding_fn=embed_fn, store=store, config=cfg)
        chunks = list(store.stream())

    assert all(ec.embedding.shape == (dim,) for ec in chunks)
