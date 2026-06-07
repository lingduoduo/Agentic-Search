"""Compatibility imports for document-index embedding."""

from src.backend.document_index.indexing import (
    DefaultIndexingEmbedder as DefaultIndexingEmbedder,
)
from src.backend.document_index.indexing import IndexingEmbedder as IndexingEmbedder
from src.backend.document_index.indexing import numpy_embedding_fn as numpy_embedding_fn
from .index_builder import embed_chunks as embed_chunks
from .index_builder import (
    embed_chunks_with_failure_handling as embed_chunks_with_failure_handling,
)


__all__ = [
    "DefaultIndexingEmbedder",
    "IndexingEmbedder",
    "embed_chunks",
    "embed_chunks_with_failure_handling",
    "numpy_embedding_fn",
]
