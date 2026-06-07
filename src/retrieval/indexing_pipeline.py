"""Compatibility imports for the public document-indexing pipeline."""

from src.backend.document_index.indexing import (
    DocumentBatchPrepareContext as DocumentBatchPrepareContext,
)
from src.backend.document_index.indexing import embed_and_stream as embed_and_stream
from src.backend.document_index.indexing import filter_documents as filter_documents
from src.backend.document_index.indexing import (
    index_document_batch as index_document_batch,
)
from src.retrieval.index_builder import run_indexing_pipeline as run_indexing_pipeline


__all__ = [
    "DocumentBatchPrepareContext",
    "embed_and_stream",
    "filter_documents",
    "index_document_batch",
    "run_indexing_pipeline",
]
