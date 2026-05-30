"""Docprocessing worker: runs the full indexing pipeline for each document batch.

Pipeline per batch:
  1. SQL upsert     — persist raw documents to AgenticSearchStore
  2. Filter         — drop empty / over-size docs and surface them as failures
  3. Chunk          — split into IndexChunk objects; ChunkingConfig.include_metadata
                      controls contextual metadata injection per chunk
  4. Embed          — embed chunks via DefaultIndexingEmbedder with per-document
                      failure isolation (failed docs are collected, not raised)
  5. Vector DB      — write EmbeddedChunks via ChunkSink with per-doc backoff retry
  6. Metadata       — write final chunk count back to the SQL store
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field

from src.connectors.models import ConnectorFailure, Document
from src.db.models import StoredDocument
from src.db.store import AgenticSearchStore
from src.retrieval.index_builder import embed_chunks_with_failure_handling
from src.retrieval.index_builder import filter_indexable_documents
from src.retrieval.indexing_heartbeat import IndexingHeartbeatInterface
from src.retrieval.models import ChunkingConfig, EmbeddingConfig
from src.servers.indexing.chunker import Chunker
from src.servers.indexing.embedder import DefaultIndexingEmbedder
from src.servers.indexing.vector_db_insertion import (
    ChunkSink,
    write_chunks_with_backoff,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocprocessingConfig:
    """Runtime configuration for the docprocessing worker."""

    max_workers: int = 2
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_db_retry_sleep_secs: float = 0.0
    connector_id: str | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "DocprocessingConfig":
        source: dict[str, str] = env if env is not None else dict(os.environ)
        return cls(
            max_workers=_env_int(source, "AGENTIC_SEARCH_NUM_DOCPROCESSING_WORKERS", 2),
        )


@dataclass
class DocprocessingResult:
    """Accumulated outcome across all processed batches."""

    batches_processed: int = 0
    total_documents: int = 0
    total_chunks: int = 0
    failures: list[ConnectorFailure] = field(default_factory=list)


class DocprocessingWorker:
    """Runs the full indexing pipeline for each document batch.

    Designed to be used as the ``processing_fn`` for ``DocfetchingWorker``::

        worker = DocprocessingWorker(store=store, chunk_sink=sink)
        fetcher = DocfetchingWorker(worker.process_batch)

    Can also be called standalone via ``run()`` for batch indexing without a
    live connector::

        result = worker.run([batch_1, batch_2, ...])

    Pass ``store=None`` to skip SQL persistence; ``chunk_sink=None`` to skip
    vector DB writes (useful in tests or offline chunking-only flows).
    """

    def __init__(
        self,
        *,
        store: AgenticSearchStore | None = None,
        chunk_sink: ChunkSink | None = None,
        embedder: DefaultIndexingEmbedder | None = None,
        config: DocprocessingConfig | None = None,
        callback: IndexingHeartbeatInterface | None = None,
    ) -> None:
        self._store = store
        self._sink = chunk_sink
        self._embedder = embedder or DefaultIndexingEmbedder()
        self._config = config or DocprocessingConfig.from_env()
        self._callback = callback

    # ------------------------------------------------------------------ #
    # Public entry points                                                  #
    # ------------------------------------------------------------------ #

    def process_batch(self, documents: list[Document]) -> DocprocessingResult:
        """Run the full pipeline for one batch.

        Thread-safe: may be called concurrently from a thread pool.
        Each failure is recorded in ``DocprocessingResult.failures`` rather
        than raised so that one bad document does not abort the whole batch.
        """
        result = DocprocessingResult(total_documents=len(documents))
        if not documents:
            return result

        # 1. SQL upsert — persist raw documents before any processing so they
        #    survive a subsequent embedding or vector DB failure.
        if self._store is not None:
            for doc in documents:
                self._store.upsert_document(
                    StoredDocument(
                        id=doc.id,
                        title=doc.title or "",
                        contents=doc.contents or "",
                        url=doc.url,
                        connector_id=self._config.connector_id,
                        metadata=dict(doc.metadata or {}),
                    )
                )

        # 2. Filter
        indexable, filter_failures = filter_indexable_documents(
            documents,
            max_document_chars=self._config.chunking.max_document_chars,
        )
        result.failures.extend(filter_failures)

        if not indexable:
            return result

        # 3. Chunk — ChunkingConfig.include_metadata injects contextual metadata
        #    text into each chunk at construction time.
        if self._callback and self._callback.should_stop():
            return result

        chunker = Chunker(self._config.chunking, callback=self._callback)
        chunks = chunker.chunk(indexable)
        result.total_chunks = len(chunks)

        if not chunks:
            return result

        # 4. Embed with per-document failure isolation: one bad document does not
        #    block the rest of the batch.
        if self._callback and self._callback.should_stop():
            return result

        embedded, embed_failures = embed_chunks_with_failure_handling(
            chunks,
            embedding_fn=self._embedder.embedding_fn,
            config=self._config.embedding,
            callback=self._callback,
        )
        result.failures.extend(embed_failures)
        failed_doc_ids = {f.document_id for f in embed_failures if f.document_id}

        if not embedded:
            return result

        # 5. Vector DB write with per-document backoff retry.
        if self._sink is not None:
            if self._callback and self._callback.should_stop():
                return result

            def _make_chunks():
                return (
                    c for c in embedded if c.chunk.document_id not in failed_doc_ids
                )

            _, write_failures = write_chunks_with_backoff(
                self._sink,
                _make_chunks,
                retry_sleep_seconds=self._config.vector_db_retry_sleep_secs,
            )
            result.failures.extend(write_failures)
            failed_doc_ids.update(
                f.document_id for f in write_failures if f.document_id
            )

        # 6. Metadata update — record chunk count on successfully indexed docs.
        if self._store is not None:
            chunks_by_doc: dict[str, int] = {}
            for chunk in chunks:
                if chunk.document_id not in failed_doc_ids:
                    chunks_by_doc[chunk.document_id] = (
                        chunks_by_doc.get(chunk.document_id, 0) + 1
                    )
            for doc in indexable:
                if doc.id in chunks_by_doc:
                    updated_metadata = {
                        **(doc.metadata or {}),
                        "indexed_chunks": chunks_by_doc[doc.id],
                    }
                    self._store.upsert_document(
                        StoredDocument(
                            id=doc.id,
                            title=doc.title or "",
                            contents=doc.contents or "",
                            url=doc.url,
                            connector_id=self._config.connector_id,
                            metadata=updated_metadata,
                        )
                    )

        return result

    def run(
        self,
        batches: Iterable[list[Document]],
    ) -> DocprocessingResult:
        """Process multiple batches concurrently.

        For standalone use — when there is no ``DocfetchingWorker`` driving the
        concurrency externally.  Results from all batches are merged and returned.
        """
        overall = DocprocessingResult()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._config.max_workers,
            thread_name_prefix="docprocessing",
        ) as pool:
            futures = [pool.submit(self.process_batch, batch) for batch in batches]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    batch_result = fut.result()
                    overall.batches_processed += 1
                    overall.total_documents += batch_result.total_documents
                    overall.total_chunks += batch_result.total_chunks
                    overall.failures.extend(batch_result.failures)
                except Exception:
                    logger.exception("Batch processing raised an unhandled exception")

        return overall


# ------------------------------------------------------------------ #
# Module-level helpers                                                #
# ------------------------------------------------------------------ #


def _env_int(env: dict[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


__all__ = [
    "DocprocessingConfig",
    "DocprocessingResult",
    "DocprocessingWorker",
]
