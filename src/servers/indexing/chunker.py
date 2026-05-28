"""Object-oriented chunking facade for the retrieval indexing pipeline."""

from __future__ import annotations

from collections.abc import Iterable

from src.connectors.models import Document
from src.retrieval.index_builder import chunk_document
from src.retrieval.index_builder import chunk_documents
from src.retrieval.index_builder import filter_indexable_documents
from src.retrieval.index_builder import generate_large_chunks
from src.retrieval.indexing_heartbeat import IndexingHeartbeatInterface
from src.retrieval.models import ChunkingConfig
from src.retrieval.models import IndexChunk


class Chunker:
    """Chunk documents with the repo's indexing configuration.

    This keeps the sampled ``Chunker`` ergonomics while delegating the actual
    chunking rules to :mod:`src.retrieval.index_builder`.
    """

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


__all__ = [
    "Chunker",
    "chunk_document",
    "chunk_documents",
    "filter_indexable_documents",
    "generate_large_chunks",
]
