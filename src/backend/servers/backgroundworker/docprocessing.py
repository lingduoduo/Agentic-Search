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

from src.backend.connectors.models import ConnectorFailure, Document
from src.backend.db.models import StoredDocument
from src.backend.db.store import AgenticSearchStore
from src.backend.document_index import ChunkSink
from src.backend.document_index import DefaultIndexingEmbedder
from src.backend.document_index import index_documents
from src.backend.document_index.index_builder import IndexingHeartbeatInterface
from src.backend.document_index.models import ChunkingConfig, EmbeddingConfig

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
            self._store.upsert_documents_bulk(
                [
                    StoredDocument(
                        id=doc.id,
                        title=doc.title or "",
                        contents=doc.contents or "",
                        url=doc.url,
                        connector_id=self._config.connector_id,
                        metadata=dict(doc.metadata or {}),
                    )
                    for doc in documents
                ]
            )

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
                    metadata={
                        **(doc.metadata or {}),
                        "indexed_chunks": chunks_by_doc[doc.id],
                    },
                )
                for doc in indexing_result.documents
                if doc.id in chunks_by_doc
            ]
            if metadata_updates:
                self._store.upsert_documents_bulk(metadata_updates)

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
