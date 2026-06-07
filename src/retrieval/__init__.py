from .context import AgentContext as AgentContext
from .context import SearchContext as SearchContext
from .context import SearchResult as SearchResult
from .client import SearchClient as SearchClient
from .client import SearchClientConfig as SearchClientConfig
from .rerank import get_reranker as get_reranker
from src.backend.document_index.retrieval import DenseRetriever as DenseRetriever
from src.backend.document_index.retrieval import (
    DenseRetrieverConfig as DenseRetrieverConfig,
)
from src.backend.document_index.retrieval import SparseRetriever as SparseRetriever
from src.backend.document_index.retrieval import (
    SparseRetrieverConfig as SparseRetrieverConfig,
)
from src.backend.document_index.index_builder import chunk_document as chunk_document
from src.backend.document_index.index_builder import chunk_documents as chunk_documents
from src.backend.document_index.index_builder import (
    deterministic_embedding_fn as deterministic_embedding_fn,
)
from src.backend.document_index.index_builder import embed_chunks as embed_chunks
from src.backend.document_index.index_builder import (
    embed_chunks_with_failure_handling as embed_chunks_with_failure_handling,
)
from src.backend.document_index.index_builder import (
    generate_large_chunks as generate_large_chunks,
)
from src.backend.document_index.index_builder import (
    run_indexing_pipeline as run_indexing_pipeline,
)
from src.backend.document_index.index_builder import (
    IndexingHeartbeatInterface as IndexingHeartbeatInterface,
)
from .models import BaseChunk as BaseChunk
from .models import ChunkEmbedding as ChunkEmbedding
from .models import ChunkEnrichmentContext as ChunkEnrichmentContext
from .models import ChunkingConfig as ChunkingConfig
from .models import DocMetadataAwareIndexChunk as DocMetadataAwareIndexChunk
from .models import DocumentAccess as DocumentAccess
from .models import EmbeddedChunk as EmbeddedChunk
from .models import EmbeddingConfig as EmbeddingConfig
from .models import EmbeddingModelDetail as EmbeddingModelDetail
from .models import EmbeddingPrecision as EmbeddingPrecision
from .models import EmbeddingProvider as EmbeddingProvider
from .models import IndexChunk as IndexChunk
from .models import IndexingBatchAdapter as IndexingBatchAdapter
from .models import IndexingPipelineConfig as IndexingPipelineConfig
from .models import IndexingPipelineResult as IndexingPipelineResult
from .models import IndexingSetting as IndexingSetting
from .models import IndexWriterConfig as IndexWriterConfig
from .models import MultipassConfig as MultipassConfig
from .models import SwitchoverType as SwitchoverType
from .models import UpdatableChunkData as UpdatableChunkData
from .hybrid_retriever import HybridRetriever as HybridRetriever
from .hybrid_retriever import HybridRetrieverConfig as HybridRetrieverConfig
from .hybrid_retriever import combine_retrieval_results as combine_retrieval_results

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
    """Lazily preserve legacy indexing imports from ``src.retrieval``."""

    if name in _INDEXING_EXPORTS:
        from src.backend.document_index import indexing

        return getattr(indexing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_retriever(
    config: DenseRetrieverConfig | SparseRetrieverConfig | HybridRetrieverConfig,
) -> DenseRetriever | SparseRetriever | HybridRetriever:
    """Instantiate the right retriever from a config object.

    HybridRetrieverConfig → HybridRetriever (fused dense + BM25).
    SparseRetrieverConfig → SparseRetriever.
    DenseRetrieverConfig with retrieval_method='bm25' → SparseRetriever.
    Anything else → DenseRetriever.
    """
    if isinstance(config, HybridRetrieverConfig):
        return HybridRetriever(config)
    if isinstance(config, SparseRetrieverConfig):
        return SparseRetriever(config)
    if config.retrieval_method.lower() == "bm25":
        return SparseRetriever(
            SparseRetrieverConfig(
                index_path=config.index_path,
                corpus_path=config.corpus_path,
                retrieval_method=config.retrieval_method,
                topk=config.topk,
            )
        )
    return DenseRetriever(config)
