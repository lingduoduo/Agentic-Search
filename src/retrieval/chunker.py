"""Compatibility imports for document-index chunking."""

from src.backend.document_index.indexing import Chunker as Chunker
from .index_builder import chunk_document as chunk_document
from .index_builder import chunk_documents as chunk_documents
from .index_builder import filter_indexable_documents as filter_indexable_documents
from .index_builder import generate_large_chunks as generate_large_chunks


__all__ = [
    "Chunker",
    "chunk_document",
    "chunk_documents",
    "filter_indexable_documents",
    "generate_large_chunks",
]
