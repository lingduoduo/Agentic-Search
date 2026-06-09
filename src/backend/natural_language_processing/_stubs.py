"""No-op stubs for tracing, metrics, and indexing heartbeat interfaces.

These replace the upstream tracing / metrics dependencies not present in this repo.
Swap in real implementations when observability tooling is available.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Generator
from enum import StrEnum
from typing import Any


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
# Embedding metrics stubs
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def track_embedding_in_progress(
    provider: Any,
    text_type: Any,
) -> Generator[None, None, None]:
    """No-op context manager for in-progress embedding tracking."""
    yield


def observe_embedding_client(
    provider: Any,
    text_type: Any,
    duration_s: float,
    num_texts: int,
    num_chars: int,
    success: bool,
) -> None:
    """No-op embedding client observation."""


# ---------------------------------------------------------------------------
# Query-embedding cache metrics stubs
# ---------------------------------------------------------------------------


class QueryEmbeddingCacheLookupOutcome(StrEnum):
    HIT = "hit"
    MISS = "miss"
    ERROR = "error"
    SKIPPED = "skipped"


class QueryEmbeddingCacheWriteOutcome(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


def observe_query_embedding_cache_lookup(
    provider: Any,
    outcome: QueryEmbeddingCacheLookupOutcome,
    count: int,
) -> None:
    """No-op cache lookup observation."""


def observe_query_embedding_cache_write(
    provider: Any,
    outcome: QueryEmbeddingCacheWriteOutcome,
    count: int,
) -> None:
    """No-op cache write observation."""


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
