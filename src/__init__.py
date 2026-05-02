"""Top-level package for Agentic-Search."""

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
    if name in __all__:
        search_module = import_module(".search", __name__)
        return getattr(search_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
