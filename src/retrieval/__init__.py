from .context import AgentContext as AgentContext
from .context import SearchContext as SearchContext
from .context import SearchResult as SearchResult
from .client import SearchClient as SearchClient
from .client import SearchClientConfig as SearchClientConfig
from .text_processor import TextProcessor as TextProcessor
from .vocabulary import Vocabulary as Vocabulary
from .rerank import get_reranker as get_reranker
from .dense_retriever import DenseRetriever as DenseRetriever
from .dense_retriever import DenseRetrieverConfig as DenseRetrieverConfig
from .index_builder import chunk_document as chunk_document
from .index_builder import chunk_documents as chunk_documents
from .index_builder import deterministic_embedding_fn as deterministic_embedding_fn
from .index_builder import embed_chunks as embed_chunks
from .index_builder import (
    embed_chunks_with_failure_handling as embed_chunks_with_failure_handling,
)
from .index_builder import generate_large_chunks as generate_large_chunks
from .index_builder import run_indexing_pipeline as run_indexing_pipeline
from .indexing_heartbeat import (
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
from .sparse_retriever import SparseRetriever as SparseRetriever
from .sparse_retriever import SparseRetrieverConfig as SparseRetrieverConfig
from .hybrid_retriever import HybridRetriever as HybridRetriever
from .hybrid_retriever import HybridRetrieverConfig as HybridRetrieverConfig
from .hybrid_retriever import combine_retrieval_results as combine_retrieval_results


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
