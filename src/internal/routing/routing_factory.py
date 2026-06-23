"""Build a Router from environment variables (default-off)."""

from __future__ import annotations

import os

from .registry import RouteRegistry
from .router import Router


def _bool(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def build_router_from_env() -> Router | None:
    """Return a Router when ROUTING_ENABLED is set, else None (zero overhead)."""
    if not _bool("ROUTING_ENABLED"):
        return None
    registry = RouteRegistry.from_env()
    llm = None
    embedder = None
    logical = _bool("ROUTING_LOGICAL")
    semantic = _bool("ROUTING_SEMANTIC")
    if logical:
        try:
            from src.internal.retrieval.service import _build_llm

            llm = _build_llm()
        except Exception:
            logical = False
    return Router(
        registry, llm=llm, embedder=embedder, logical=logical, semantic=semantic
    )
