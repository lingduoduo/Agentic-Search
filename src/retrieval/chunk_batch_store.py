"""Compatibility import for document-index batch storage."""

from src.backend.document_index.indexing import ChunkBatchStore as ChunkBatchStore


__all__ = ["ChunkBatchStore"]
