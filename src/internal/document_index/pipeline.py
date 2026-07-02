"""End-to-end index-building pipeline: chunk, embed, and write index artifacts
for a batch of connector documents.

Orchestrates the chunking, embedding, and faiss_io modules; kept separate so the
top-level flow reads without the implementation detail of each stage.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.internal.connectors.models import Document
from src.internal.document_index._common import (
    IndexingHeartbeatInterface,
    _raise_if_indexing_stopped,
    _report_indexing_progress,
)
from src.internal.document_index.chunking import (
    chunk_documents,
    filter_indexable_documents,
)
from src.internal.document_index.embedding import EmbeddingFn, embed_chunks
from src.internal.document_index.faiss_io import (
    write_corpus_jsonl,
    write_embeddings_memmap,
    write_faiss_index,
)
from src.internal.document_index.models import (
    IndexingPipelineConfig,
    IndexingPipelineResult,
)


def run_indexing_pipeline(
    documents: Iterable[Document],
    *,
    config: IndexingPipelineConfig,
    embedding_fn: EmbeddingFn,
    callback: IndexingHeartbeatInterface | None = None,
) -> IndexingPipelineResult:
    """Chunk, embed, and write index artifacts for connector documents."""

    config.validate()
    docs = list(documents)
    indexable_docs, failures = filter_indexable_documents(
        docs,
        max_document_chars=config.chunking.max_document_chars,
    )
    _raise_if_indexing_stopped(callback, "run_indexing_pipeline")
    chunks = chunk_documents(indexable_docs, config.chunking, callback=callback)
    if not chunks:
        raise ValueError("No non-empty chunks were produced.")

    save_dir = Path(config.writer.save_dir)
    corpus_path = save_dir / config.writer.corpus_filename
    embedding_path = save_dir / config.writer.embedding_filename
    index_path = save_dir / config.writer.index_filename

    _raise_if_indexing_stopped(callback, "write_corpus_jsonl")
    write_corpus_jsonl(chunks, corpus_path)
    _report_indexing_progress(callback, "write_corpus_jsonl", len(chunks))
    embedded_chunks = embed_chunks(
        chunks,
        embedding_fn=embedding_fn,
        config=config.embedding,
        callback=callback,
    )
    _raise_if_indexing_stopped(callback, "write_embeddings_memmap")
    write_embeddings_memmap(embedded_chunks, embedding_path)
    _report_indexing_progress(callback, "write_embeddings_memmap", len(embedded_chunks))

    written_index_path = None
    if config.writer.write_faiss:
        _raise_if_indexing_stopped(callback, "write_faiss_index")
        written_index_path = write_faiss_index(
            embedded_chunks,
            index_path,
            faiss_type=config.writer.faiss_type,
            hnsw_ef_construction=config.writer.hnsw_ef_construction,
            hnsw_ef_search=config.writer.hnsw_ef_search,
        )
        _report_indexing_progress(callback, "write_faiss_index", len(embedded_chunks))

    return IndexingPipelineResult(
        total_documents=len(docs),
        total_chunks=len(chunks),
        corpus_path=corpus_path,
        embedding_path=embedding_path,
        index_path=written_index_path,
        failures=failures,
    )
