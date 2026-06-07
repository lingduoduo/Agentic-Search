"""Compatibility imports for the document-index insertion helpers."""

from src.backend.document_index.indexing import ChunkSink as ChunkSink
from src.backend.document_index.indexing import (
    write_chunks_to_vector_db_with_backoff as write_chunks_to_vector_db_with_backoff,
)
from src.backend.document_index.indexing import (
    write_chunks_with_backoff as write_chunks_with_backoff,
)

__all__ = [
    "ChunkSink",
    "write_chunks_to_vector_db_with_backoff",
    "write_chunks_with_backoff",
]
