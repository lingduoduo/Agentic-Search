"""Document-index backends, text handling, and indexing entry points."""

_BUILDER_EXPORTS = {
    "IndexBuilder",
    "IndexBuilderConfig",
    "IndexingHeartbeatInterface",
}

_INDEXING_EXPORTS = {
    "ChunkBatchStore",
    "ChunkSink",
    "Chunker",
    "DefaultIndexingEmbedder",
    "DocumentBatchPrepareContext",
    "DocumentIndexingResult",
    "IndexingEmbedder",
    "embed_and_stream",
    "filter_documents",
    "index_document_batch",
    "index_documents",
    "write_chunks_with_backoff",
}


def __getattr__(name: str):
    if name in _BUILDER_EXPORTS:
        from . import index_builder

        return getattr(index_builder, name)
    if name in _INDEXING_EXPORTS:
        from . import indexing

        return getattr(indexing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
