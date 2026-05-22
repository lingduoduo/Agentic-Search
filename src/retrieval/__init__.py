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
from .index_builder import ChunkingConfig as ChunkingConfig
from .index_builder import EmbeddedChunk as EmbeddedChunk
from .index_builder import EmbeddingConfig as EmbeddingConfig
from .index_builder import IndexChunk as IndexChunk
from .index_builder import IndexingPipelineConfig as IndexingPipelineConfig
from .index_builder import IndexingPipelineResult as IndexingPipelineResult
from .index_builder import IndexWriterConfig as IndexWriterConfig
from .index_builder import chunk_document as chunk_document
from .index_builder import chunk_documents as chunk_documents
from .index_builder import deterministic_embedding_fn as deterministic_embedding_fn
from .index_builder import embed_chunks as embed_chunks
from .index_builder import run_indexing_pipeline as run_indexing_pipeline
from .sparse_retriever import SparseRetriever as SparseRetriever
from .sparse_retriever import SparseRetrieverConfig as SparseRetrieverConfig


def build_retriever(
    config: DenseRetrieverConfig | SparseRetrieverConfig,
) -> DenseRetriever | SparseRetriever:
    """Instantiate the right retriever from a config object.

    Routes SparseRetrieverConfig (or any DenseRetrieverConfig whose
    retrieval_method is 'bm25') to SparseRetriever; everything else to
    DenseRetriever.
    """
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
