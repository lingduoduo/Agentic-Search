"""Repo-native indexing helpers and compatibility facade."""

from .chunk_batch_store import ChunkBatchStore
from .chunker import Chunker
from .embedder import DefaultIndexingEmbedder
from .embedder import IndexingEmbedder
from .embedder import embed_chunks_with_failure_handling
from .indexing_pipeline import DocumentBatchPrepareContext
from .indexing_pipeline import embed_and_stream
from .indexing_pipeline import filter_documents
from .indexing_pipeline import index_document_batch
from .indexing_pipeline import run_indexing_pipeline

__all__ = [
    "ChunkBatchStore",
    "Chunker",
    "DefaultIndexingEmbedder",
    "DocumentBatchPrepareContext",
    "IndexingEmbedder",
    "embed_and_stream",
    "embed_chunks_with_failure_handling",
    "filter_documents",
    "index_document_batch",
    "run_indexing_pipeline",
]
