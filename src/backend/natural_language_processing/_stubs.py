"""Tracing stubs and re-exports of real metrics implementations.

Tracing (LLM call spans) remains a no-op until a tracing backend is wired in.
Embedding and cache metrics delegate to the real Prometheus implementations in
src.backend.metrics.embedding.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Generator
from enum import StrEnum

# Real Prometheus implementations — re-exported so callers only need to import
# from this module.
from src.backend.metrics.embedding import QueryEmbeddingCacheLookupOutcome  # noqa: F401
from src.backend.metrics.embedding import QueryEmbeddingCacheWriteOutcome  # noqa: F401
from src.backend.metrics.embedding import observe_embedding_client  # noqa: F401
from src.backend.metrics.embedding import observe_query_embedding_cache_lookup  # noqa: F401
from src.backend.metrics.embedding import observe_query_embedding_cache_write  # noqa: F401
from src.backend.metrics.embedding import track_embedding_in_progress  # noqa: F401


# ---------------------------------------------------------------------------
# Tracing stubs
# ---------------------------------------------------------------------------


class LLMFlow(StrEnum):
    EMBED_PASSAGE = "embed_passage"
    EMBED_QUERY = "embed_query"
    RERANK = "rerank"
    INTENT_CLASSIFICATION = "intent_classification"


@contextlib.contextmanager
def traced_llm_call(
    *,
    flow: LLMFlow,
    model: str,
    provider: str,
    extra_config: dict[str, str] | None = None,
) -> Generator[None, None, None]:
    """No-op context manager standing in for LLM call tracing."""
    yield


# ---------------------------------------------------------------------------
# Indexing heartbeat interface
# ---------------------------------------------------------------------------


class IndexingHeartbeatInterface(ABC):
    """Abstract interface for signalling stop requests during indexing."""

    @abstractmethod
    def should_stop(self) -> bool: ...

    @abstractmethod
    def heartbeat(self) -> None: ...


# ---------------------------------------------------------------------------
# AWS key helper
# ---------------------------------------------------------------------------


def pass_aws_key(api_key: str) -> tuple[str, str, str]:
    """Parse a combined AWS credential string of the form 'key_id:secret:region'.

    Returns (aws_access_key_id, aws_secret_access_key, region_name).
    """
    parts = api_key.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            "AWS api_key must be formatted as 'access_key_id:secret_access_key:region'"
        )
    return parts[0], parts[1], parts[2]
