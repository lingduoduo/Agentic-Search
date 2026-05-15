"""Search package exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "OnlineSearchConfig",
    "OnlineSearchEngine",
    "SerpSearchConfig",
    "SerpSearchEngine",
    "IndexBuilder",
    "IndexBuilderConfig",
    "DenseRetriever",
    "DenseRetrieverConfig",
    "SparseRetriever",
    "SparseRetrieverConfig",
    "RetrievalServerConfig",
    "RerankerConfig",
    "RetrievalRerankConfig",
    "create_base_app",
    "Vocabulary",
    "SOS_token",
    "EOS_token",
    "MAX_LENGTH",
    "normalize_text",
    "normalize_document",
    "tokenize_text",
    "tokenize_document",
    "build_vocabulary_from_sequences",
    "extract_keywords",
    "TextProcessor",
]

_LAZY_EXPORTS: dict[str, str] = {
    "OnlineSearchConfig": ".google_search_server",
    "OnlineSearchEngine": ".google_search_server",
    "SerpSearchConfig": ".serp_search_server",
    "SerpSearchEngine": ".serp_search_server",
    "IndexBuilder": ".index_builder",
    "IndexBuilderConfig": ".index_builder",
    "DenseRetriever": ".retrieval",
    "DenseRetrieverConfig": ".retrieval",
    "SparseRetriever": ".sparse_retriever",
    "SparseRetrieverConfig": ".sparse_retriever",
    "RetrievalServerConfig": ".retrieval_server",
    "RerankerConfig": ".rerank",
    "RetrievalRerankConfig": ".retrieval_rerank_server",
    "create_base_app": ".search_app",
    "TextProcessor": ".text_processor",
    "Vocabulary": ".vocabulary",
    "SOS_token": ".vocabulary",
    "EOS_token": ".vocabulary",
    "MAX_LENGTH": ".vocabulary",
    "normalize_text": ".vocabulary",
    "normalize_document": ".vocabulary",
    "tokenize_text": ".vocabulary",
    "tokenize_document": ".vocabulary",
    "build_vocabulary_from_sequences": ".vocabulary",
    "extract_keywords": ".vocabulary",
}


def __getattr__(name: str) -> Any:
    try:
        module_path = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_path, __name__), name)
    globals()[name] = value
    return value
