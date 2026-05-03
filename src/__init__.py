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
    "Vocabulary",
    "SOS_token",
    "EOS_token",
    "MAX_LENGTH",
    "normalize_text",
    "tokenize_text",
    "build_vocabulary_from_sequences",
    "extract_keywords",
    "GenerationConfig",
    "LLMGenerationManager",
    "TensorConfig",
    "TensorHelper",
]


def __getattr__(name: str) -> Any:
    if name in {
        "GenerationConfig",
        "LLMGenerationManager",
        "TensorConfig",
        "TensorHelper",
    }:
        llm_agent_module = import_module(".llm_agent", __name__)
        return getattr(llm_agent_module, name)
    if name in __all__:
        search_module = import_module(".search", __name__)
        return getattr(search_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
