"""Back-compat facade for the index-building subsystem.

The implementation was split into cohesive modules:
  - ``_common``    — optional-dependency guards + indexing heartbeat
  - ``chunking``   — document chunking + text splitting
  - ``embedding``  — model loading, pooling, chunk embedding
  - ``faiss_io``   — FAISS/memmap/JSONL writers + corpus I/O
  - ``pipeline``   — ``run_indexing_pipeline`` orchestration
  - ``cli``        — the ``python -m`` index-builder command-line tool

This module re-exports the public surface so existing imports such as
``from src.internal.document_index.index_builder import chunk_documents`` and
``python -m src.internal.document_index.index_builder`` keep working unchanged.
"""

# This module intentionally re-exports names for backward compatibility.
# ruff: noqa: F401

from __future__ import annotations

from src.internal.document_index._common import (
    IndexingHeartbeatInterface,
    _raise_if_indexing_stopped,
    _report_indexing_progress,
    _require_faiss,
    _require_torch,
    _require_transformers,
    _require_tqdm,
)
from src.internal.document_index.chunking import (
    RETURN_SEPARATOR,
    SECTION_SEPARATOR,
    _combine_index_chunks,
    _extract_blurb,
    _make_mini_chunk_texts,
    _metadata_suffix_for_index,
    _overlap_tail,
    _split_paragraphs,
    _split_sentences_in_paragraph,
    _split_text,
    _split_text_paragraphs,
    _split_token_window,
    _token_count,
    _tokenize_for_chunking,
    chunk_document,
    chunk_documents,
    filter_indexable_documents,
    generate_large_chunks,
)
from src.internal.document_index.cli import (
    IndexBuilder,
    IndexBuilderConfig,
    main,
    parse_args,
)
from src.internal.document_index.embedding import (
    MODEL2POOLING,
    EmbeddingFn,
    _apply_text_prefix,
    _batched,
    _coerce_embedding_matrix,
    _embed_chunk_batch,
    _embed_texts,
    _encode_batch,
    _get_title_embedding,
    _normalize_embedding_rows,
    deterministic_embedding_fn,
    embed_chunks,
    embed_chunks_with_failure_handling,
    load_model,
    pooling,
    prepare_texts,
    resolve_pooling_method,
)
from src.internal.document_index.faiss_io import (
    _Corpus,
    dump_connector_to_jsonl,
    load_corpus,
    load_corpus_from_connector,
    set_hnsw_ef_construction,
    set_hnsw_ef_search,
    write_corpus_jsonl,
    write_dense_faiss_index,
    write_embeddings_memmap,
    write_faiss_index,
)
from src.internal.document_index.models import (
    ChunkingConfig,
    EmbeddedChunk,
    EmbeddingConfig,
    IndexChunk,
    IndexingPipelineConfig,
    IndexingPipelineResult,
    IndexWriterConfig,
)
from src.internal.document_index.pipeline import run_indexing_pipeline

__all__ = [
    "ChunkingConfig",
    "EmbeddedChunk",
    "EmbeddingConfig",
    "IndexChunk",
    "IndexingHeartbeatInterface",
    "IndexingPipelineConfig",
    "IndexingPipelineResult",
    "IndexWriterConfig",
    "_split_paragraphs",
    "_split_sentences_in_paragraph",
    "chunk_document",
    "chunk_documents",
    "deterministic_embedding_fn",
    "embed_chunks",
    "embed_chunks_with_failure_handling",
    "filter_indexable_documents",
    "generate_large_chunks",
    "prepare_texts",
    "run_indexing_pipeline",
    "write_faiss_index",
]


if __name__ == "__main__":
    main()
