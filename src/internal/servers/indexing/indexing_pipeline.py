"""Streaming batch indexing pipeline facade.

Extends the document_index.index_builder pipeline with:
- Per-batch streaming results (no all-at-once memory load)
- Integrated ChunkBatchStore for temp-disk batch management
- Per-document prefilter failure reporting
- Cancellation via IndexingHeartbeatInterface
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from src.internal.connectors.models import ConnectorFailure, Document
from src.internal.document_index.index_builder import (
    EmbeddingFn,
    IndexingHeartbeatInterface,
    chunk_documents,
    embed_chunks_with_failure_handling,
    filter_indexable_documents,
)
from src.internal.document_index.models import ChunkingConfig, EmbeddingConfig
from src.internal.servers.indexing.chunk_batch_store import ChunkBatchStore


@dataclass(frozen=True)
class StreamingIndexingConfig:
    """Configuration for the streaming batch indexing pipeline."""

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    # Number of documents per batch.  Smaller values reduce peak memory;
    # larger values amortize embedding model overhead.
    docs_per_batch: int = 32


@dataclass
class BatchIndexingResult:
    """Result for a single batch yielded by stream_indexing_pipeline."""

    batch_idx: int
    docs_in_batch: int
    chunks_produced: int
    chunks_embedded: int
    failures: list[ConnectorFailure]


@dataclass
class StreamingPipelineResult:
    """Aggregate result returned by run_indexing_pipeline."""

    total_documents: int
    total_chunks: int
    total_embedded: int
    failures: list[ConnectorFailure]


def _batched(docs: list[Document], size: int) -> Iterator[list[Document]]:
    for i in range(0, len(docs), size):
        yield docs[i : i + size]


def stream_indexing_pipeline(
    documents: Iterable[Document],
    *,
    embedding_fn: EmbeddingFn,
    store: ChunkBatchStore,
    config: StreamingIndexingConfig | None = None,
    callback: IndexingHeartbeatInterface | None = None,
) -> Iterator[BatchIndexingResult]:
    """Chunk, embed, and store documents batch by batch, yielding a result per batch.

    Caller usage::

        with ChunkBatchStore() as store:
            for batch_result in stream_indexing_pipeline(docs, embedding_fn=fn, store=store):
                print(f"Batch {batch_result.batch_idx}: {batch_result.chunks_embedded} chunks")
            for chunk in store.stream():
                vector_db.insert(chunk)

    Prefilter failures (duplicate ids, empty content, oversized documents) and
    embedding failures are reported in ``BatchIndexingResult.failures`` rather
    than raised as exceptions.  The pipeline continues with the remaining
    documents.

    A stop signal from *callback* ends iteration cleanly — no partial batch is
    written to *store*.

    Args:
        documents: Documents to index.  Consumed lazily in batches.
        embedding_fn: ``(list[str]) -> np.ndarray`` — maps text to embeddings.
        store: Open ChunkBatchStore context.  Embedded batches are saved here.
        config: Chunking, embedding, and batch-size settings.
        callback: Optional heartbeat for progress reporting and cancellation.

    Yields:
        One :class:`BatchIndexingResult` per batch after it is fully embedded
        and saved to *store*.
    """
    cfg = config or StreamingIndexingConfig()
    all_docs = list(documents)

    for batch_idx, batch_docs in enumerate(_batched(all_docs, cfg.docs_per_batch)):
        if callback and callback.should_stop():
            return

        indexable, prefilter_failures = filter_indexable_documents(
            batch_docs,
            max_document_chars=cfg.chunking.max_document_chars,
        )

        if not indexable:
            yield BatchIndexingResult(
                batch_idx=batch_idx,
                docs_in_batch=len(batch_docs),
                chunks_produced=0,
                chunks_embedded=0,
                failures=prefilter_failures,
            )
            continue

        try:
            chunks = chunk_documents(indexable, cfg.chunking, callback=callback)
        except RuntimeError as exc:
            if "stop signal detected" in str(exc):
                return
            raise

        if callback and callback.should_stop():
            return

        embedded, embed_failures = embed_chunks_with_failure_handling(
            chunks,
            embedding_fn=embedding_fn,
            config=cfg.embedding,
            callback=callback,
        )

        if embed_failures:
            failed_ids = {f.document_id for f in embed_failures if f.document_id}
            store.scrub_failed_docs(failed_ids)

        if embedded:
            store.save(embedded, batch_idx=batch_idx)

        if callback:
            callback.progress("batch_embedded", len(embedded))

        yield BatchIndexingResult(
            batch_idx=batch_idx,
            docs_in_batch=len(batch_docs),
            chunks_produced=len(chunks),
            chunks_embedded=len(embedded),
            failures=prefilter_failures + embed_failures,
        )


def run_indexing_pipeline(
    documents: Iterable[Document],
    *,
    embedding_fn: EmbeddingFn,
    store: ChunkBatchStore,
    config: StreamingIndexingConfig | None = None,
    callback: IndexingHeartbeatInterface | None = None,
) -> StreamingPipelineResult:
    """Chunk, embed, and store all documents; return an aggregate summary.

    Wraps :func:`stream_indexing_pipeline` and collects all batch results.
    After this call, use ``store.stream()`` to iterate the embedded chunks.

    Args:
        documents: Documents to index.
        embedding_fn: ``(list[str]) -> np.ndarray`` — maps text to embeddings.
        store: Open ChunkBatchStore context.
        config: Chunking, embedding, and batch-size settings.
        callback: Optional heartbeat for progress reporting and cancellation.

    Returns:
        :class:`StreamingPipelineResult` with aggregate counts and all failures.
    """
    total_docs = total_chunks = total_embedded = 0
    all_failures: list[ConnectorFailure] = []

    for batch_result in stream_indexing_pipeline(
        documents,
        embedding_fn=embedding_fn,
        store=store,
        config=config,
        callback=callback,
    ):
        total_docs += batch_result.docs_in_batch
        total_chunks += batch_result.chunks_produced
        total_embedded += batch_result.chunks_embedded
        all_failures.extend(batch_result.failures)

    return StreamingPipelineResult(
        total_documents=total_docs,
        total_chunks=total_chunks,
        total_embedded=total_embedded,
        failures=all_failures,
    )
