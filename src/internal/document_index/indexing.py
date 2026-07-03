"""Public document-indexing pipeline.

This module is the single entry point for turning connector documents into
embedded chunks and writing them to an index sink. Query-time retrieval
lives in :mod:`src.internal.document_index.models`.
"""

from __future__ import annotations

import pickle
import shutil
import tempfile
import time
from collections.abc import Callable, Generator, Iterable
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from typing import Protocol, TypeVar

import numpy as np

from src.internal.connectors.models import ConnectorFailure, Document
from src.internal.document_index._common import IndexingHeartbeatInterface
from src.internal.document_index.chunking import (
    chunk_document,
    chunk_documents,
    filter_indexable_documents,
)
from src.internal.document_index.embedding import (
    EmbeddingFn,
    deterministic_embedding_fn,
    embed_chunks,
    embed_chunks_with_failure_handling,
)
from src.internal.document_index.pipeline import run_indexing_pipeline
from src.internal.document_index.models import (
    ChunkingConfig,
    EmbeddedChunk,
    EmbeddingConfig,
    IndexChunk,
    IndexingPipelineConfig,
    IndexingPipelineResult,
    IndexWriterConfig,
)

T = TypeVar("T")


class Chunker:
    """Chunk connector documents with the configured indexing rules."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        *,
        callback: IndexingHeartbeatInterface | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self.callback = callback

    def chunk_document(self, document: Document) -> list[IndexChunk]:
        return chunk_document(document, self.config)

    def chunk(self, documents: Iterable[Document]) -> list[IndexChunk]:
        return chunk_documents(documents, self.config, callback=self.callback)


class IndexingEmbedder(Protocol):
    """Converts index chunks into embedded chunks."""

    def embed_chunks(self, chunks: list[IndexChunk]) -> list[EmbeddedChunk]: ...


class DefaultIndexingEmbedder:
    """Embedding wrapper used by document-indexing flows."""

    def __init__(
        self,
        embedding_fn: EmbeddingFn | None = None,
        config: EmbeddingConfig | None = None,
        *,
        callback: IndexingHeartbeatInterface | None = None,
    ) -> None:
        self.embedding_fn = embedding_fn or deterministic_embedding_fn()
        self.config = config or EmbeddingConfig()
        self.callback = callback

    def embed_chunks(self, chunks: list[IndexChunk]) -> list[EmbeddedChunk]:
        return embed_chunks(
            chunks,
            embedding_fn=self.embedding_fn,
            config=self.config,
            callback=self.callback,
        )


def numpy_embedding_fn(vectors: list[list[float]]) -> EmbeddingFn:
    """Build a deterministic test embedder from fixed vectors."""

    matrix = np.asarray(vectors, dtype=np.float32)

    def embed(texts: list[str]) -> np.ndarray:
        if len(texts) > len(matrix):
            raise ValueError("Not enough fixed vectors for requested texts.")
        return matrix[: len(texts)]

    return embed


class ChunkBatchStore:
    """Temporary on-disk storage for embedded chunk batches."""

    _EXT = ".pkl"

    def __init__(self) -> None:
        self._tmpdir: Path | None = None

    def __enter__(self) -> "ChunkBatchStore":
        self._tmpdir = Path(tempfile.mkdtemp(prefix="agentic_search_embeddings_"))
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    @property
    def _dir(self) -> Path:
        if self._tmpdir is None:
            raise RuntimeError("ChunkBatchStore used outside context manager.")
        return self._tmpdir

    def save(self, chunks: list[EmbeddedChunk], batch_idx: int) -> None:
        with (self._dir / f"batch_{batch_idx}{self._EXT}").open("wb") as handle:
            pickle.dump(chunks, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def stream(self) -> Iterator[EmbeddedChunk]:
        for batch_file in self._batch_files():
            yield from self._load(batch_file)

    def scrub_failed_docs(self, failed_doc_ids: set[str]) -> None:
        for batch_file in self._batch_files():
            batch_chunks = self._load(batch_file)
            cleaned = [
                chunk
                for chunk in batch_chunks
                if chunk.chunk.document_id not in failed_doc_ids
            ]
            if len(cleaned) != len(batch_chunks):
                with batch_file.open("wb") as handle:
                    pickle.dump(cleaned, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load(self, batch_file: Path) -> list[EmbeddedChunk]:
        with batch_file.open("rb") as handle:
            return pickle.load(handle)  # noqa: S301 - reads files written by save().

    def _batch_files(self) -> list[Path]:
        return sorted(
            self._dir.glob(f"batch_*{self._EXT}"),
            key=lambda path: int(path.stem.removeprefix("batch_")),
        )


class ChunkSink(Protocol[T]):
    """Minimal index interface required by the indexing pipeline."""

    def index(self, chunks: Iterable[EmbeddedChunk]) -> list[T]: ...


@dataclass
class DocumentIndexingResult:
    """Outcome of filtering, chunking, embedding, and optional index writing."""

    documents: list[Document] = field(default_factory=list)
    chunks: list[IndexChunk] = field(default_factory=list)
    embedded_chunks: list[EmbeddedChunk] = field(default_factory=list)
    records: list[object] = field(default_factory=list)
    failures: list[ConnectorFailure] = field(default_factory=list)
    stopped: bool = False

    @property
    def successful_chunk_counts(self) -> dict[str, int]:
        if self.stopped:
            return {}
        failed_document_ids = {
            failure.document_id for failure in self.failures if failure.document_id
        }
        counts: dict[str, int] = {}
        for chunk in self.chunks:
            if chunk.document_id not in failed_document_ids:
                counts[chunk.document_id] = counts.get(chunk.document_id, 0) + 1
        return counts


@dataclass(frozen=True)
class DocumentBatchPrepareContext:
    """Legacy filtered-batch container retained for import compatibility."""

    indexable_docs: list[Document]
    failures: list[ConnectorFailure] = field(default_factory=list)


def write_chunks_with_backoff(
    sink: ChunkSink[T],
    make_chunks: Callable[[], Iterable[EmbeddedChunk]],
    *,
    retry_sleep_seconds: float = 0.0,
) -> tuple[list[T], list[ConnectorFailure]]:
    """Write all chunks, then isolate per-document failures on retry."""

    try:
        return sink.index(make_chunks()), []
    except Exception:
        if retry_sleep_seconds:
            time.sleep(retry_sleep_seconds)

    records: list[T] = []
    failures: list[ConnectorFailure] = []
    seen_doc_ids: set[str] = set()

    for document_id, doc_chunks in groupby(
        make_chunks(), key=lambda chunk: chunk.chunk.document_id
    ):
        if document_id in seen_doc_ids:
            raise RuntimeError(
                "Chunks must be grouped by document before per-document retry."
            )
        seen_doc_ids.add(document_id)
        doc_chunk_list = list(doc_chunks)
        try:
            records.extend(sink.index(doc_chunk_list))
        except Exception as exc:
            failures.append(
                ConnectorFailure(
                    document_id=document_id,
                    message=str(exc),
                    exception_type=type(exc).__name__,
                    metadata={
                        "chunk_ids": [chunk.chunk.id for chunk in doc_chunk_list]
                    },
                )
            )

    return records, failures


def index_documents(
    documents: Iterable[Document],
    *,
    sink: ChunkSink[object] | None = None,
    chunking: ChunkingConfig | None = None,
    embedding: EmbeddingConfig | None = None,
    embedder: DefaultIndexingEmbedder | None = None,
    callback: IndexingHeartbeatInterface | None = None,
    retry_sleep_seconds: float = 0.0,
) -> DocumentIndexingResult:
    """Run the complete in-memory indexing pipeline for a document batch."""

    chunking_config = chunking or ChunkingConfig()
    effective_embedder = embedder or DefaultIndexingEmbedder(callback=callback)
    embedding_config = embedding or effective_embedder.config
    indexable_documents, failures = filter_indexable_documents(
        documents,
        max_document_chars=chunking_config.max_document_chars,
    )
    result = DocumentIndexingResult(
        documents=indexable_documents,
        failures=list(failures),
    )
    if not indexable_documents:
        return result
    if _should_stop(callback):
        result.stopped = True
        return result

    result.chunks = Chunker(chunking_config, callback=callback).chunk(
        indexable_documents
    )
    if not result.chunks:
        return result
    if _should_stop(callback):
        result.stopped = True
        return result

    result.embedded_chunks, embedding_failures = embed_chunks_with_failure_handling(
        result.chunks,
        embedding_fn=effective_embedder.embedding_fn,
        config=embedding_config,
        callback=callback,
    )
    result.failures.extend(embedding_failures)
    if sink is None or not result.embedded_chunks:
        return result
    if _should_stop(callback):
        result.stopped = True
        return result

    failed_document_ids = {
        failure.document_id for failure in result.failures if failure.document_id
    }
    records, write_failures = write_chunks_with_backoff(
        sink,
        lambda: (
            chunk
            for chunk in result.embedded_chunks
            if chunk.chunk.document_id not in failed_document_ids
        ),
        retry_sleep_seconds=retry_sleep_seconds,
    )
    result.records.extend(records)
    result.failures.extend(write_failures)
    return result


def filter_documents(
    document_batch: Iterable[Document],
    *,
    max_document_chars: int | None = None,
) -> tuple[list[Document], list[ConnectorFailure]]:
    """Filter invalid documents before indexing."""

    return filter_indexable_documents(
        document_batch,
        max_document_chars=max_document_chars,
    )


def index_document_batch(
    document_batch: Iterable[Document],
    *,
    save_dir: str,
    chunking: ChunkingConfig | None = None,
    embedding: EmbeddingConfig | None = None,
    writer: IndexWriterConfig | None = None,
    embedding_fn: EmbeddingFn | None = None,
    callback: IndexingHeartbeatInterface | None = None,
) -> IndexingPipelineResult:
    """Build local retrieval artifacts for one document batch."""

    chunking_config = chunking or ChunkingConfig()
    embedding_config = embedding or EmbeddingConfig()
    return run_indexing_pipeline(
        document_batch,
        config=IndexingPipelineConfig(
            chunking=chunking_config,
            embedding=embedding_config,
            writer=writer or IndexWriterConfig(save_dir=save_dir),
        ),
        embedding_fn=embedding_fn or deterministic_embedding_fn(),
        callback=callback,
    )


@contextmanager
def embed_and_stream(
    documents: Iterable[Document],
    *,
    chunking: ChunkingConfig | None = None,
    embedder: DefaultIndexingEmbedder | None = None,
    callback: IndexingHeartbeatInterface | None = None,
) -> Generator[tuple[list[ConnectorFailure], ChunkBatchStore], None, None]:
    """Embed documents into temporary batches and stream them incrementally."""

    chunking_config = chunking or ChunkingConfig()
    effective_embedder = embedder or DefaultIndexingEmbedder(callback=callback)
    indexable_documents, failures = filter_indexable_documents(
        documents,
        max_document_chars=chunking_config.max_document_chars,
    )
    chunks = Chunker(chunking_config, callback=callback).chunk(indexable_documents)
    with ChunkBatchStore() as store:
        batch_size = effective_embedder.config.batch_size
        for batch_idx, start in enumerate(range(0, len(chunks), batch_size)):
            embedded_chunks = effective_embedder.embed_chunks(
                chunks[start : start + batch_size]
            )
            if embedded_chunks:
                store.save(embedded_chunks, batch_idx=batch_idx)
        yield failures, store


def _should_stop(callback: IndexingHeartbeatInterface | None) -> bool:
    return bool(callback and callback.should_stop())


__all__ = [
    "ChunkBatchStore",
    "ChunkSink",
    "Chunker",
    "DefaultIndexingEmbedder",
    "DocumentBatchPrepareContext",
    "DocumentIndexingResult",
    "IndexingEmbedder",
    "embed_and_stream",
    "filter_documents",
    "index_document_batch",
    "index_documents",
    "numpy_embedding_fn",
    "write_chunks_with_backoff",
]
