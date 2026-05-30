"""Lightweight indexing pipeline entrypoints."""

from __future__ import annotations

from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field

from src.backend.connectors.models import ConnectorFailure
from src.backend.connectors.models import Document
from .index_builder import EmbeddingFn
from .index_builder import chunk_documents
from .index_builder import deterministic_embedding_fn
from .index_builder import filter_indexable_documents
from .index_builder import run_indexing_pipeline
from .indexing_heartbeat import IndexingHeartbeatInterface
from .models import ChunkingConfig
from .models import EmbeddedChunk
from .models import EmbeddingConfig
from .models import IndexingPipelineConfig
from .models import IndexingPipelineResult
from .models import IndexWriterConfig

from .chunk_batch_store import ChunkBatchStore
from .embedder import DefaultIndexingEmbedder


@dataclass(frozen=True)
class DocumentBatchPrepareContext:
    """Filtered document batch plus prefilter failures."""

    indexable_docs: list[Document]
    failures: list[ConnectorFailure] = field(default_factory=list)


def filter_documents(
    document_batch: Iterable[Document],
    *,
    max_document_chars: int | None = None,
) -> tuple[list[Document], list[ConnectorFailure]]:
    """Compatibility wrapper for document prefiltering."""

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
    """Run chunking, embedding, and artifact writing for one document batch."""

    chunking_config = chunking or ChunkingConfig()
    embedding_config = embedding or EmbeddingConfig()
    writer_config = writer or IndexWriterConfig(save_dir=save_dir)
    return run_indexing_pipeline(
        document_batch,
        config=IndexingPipelineConfig(
            chunking=chunking_config,
            embedding=embedding_config,
            writer=writer_config,
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
    """Chunk and embed documents to temporary batch files, then stream them."""

    chunking_config = chunking or ChunkingConfig()
    indexable_docs, failures = filter_indexable_documents(
        documents,
        max_document_chars=chunking_config.max_document_chars,
    )
    chunks = chunk_documents(indexable_docs, chunking_config, callback=callback)
    effective_embedder = embedder or DefaultIndexingEmbedder(callback=callback)

    with ChunkBatchStore() as store:
        embedded_chunks: list[EmbeddedChunk] = effective_embedder.embed_chunks(chunks)
        store.save(embedded_chunks, batch_idx=0)
        yield failures, store


__all__ = [
    "DocumentBatchPrepareContext",
    "embed_and_stream",
    "filter_documents",
    "index_document_batch",
    "run_indexing_pipeline",
]
