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
    "RetrievalServerConfig",
    "RerankerConfig",
    "RetrievalRerankConfig",
    "create_base_app",
]


def __getattr__(name: str) -> Any:
    if name in {"OnlineSearchConfig", "OnlineSearchEngine"}:
        module = import_module(".google_search_server", __name__)
        return getattr(module, name)
    if name in {"SerpSearchConfig", "SerpSearchEngine"}:
        module = import_module(".serp_search_server", __name__)
        return getattr(module, name)
    if name in {"IndexBuilder", "IndexBuilderConfig"}:
        module = import_module(".index_builder", __name__)
        return getattr(module, name)
    if name in {"DenseRetriever", "DenseRetrieverConfig"}:
        module = import_module(".retrieval", __name__)
        return getattr(module, name)
    if name == "RetrievalServerConfig":
        module = import_module(".retrieval_server", __name__)
        return getattr(module, name)
    if name == "RerankerConfig":
        module = import_module(".rerank", __name__)
        return getattr(module, name)
    if name == "RetrievalRerankConfig":
        module = import_module(".retrieval_rerank_server", __name__)
        return getattr(module, name)
    if name == "create_base_app":
        module = import_module(".search_app", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
