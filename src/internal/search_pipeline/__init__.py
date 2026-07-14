"""Internal building blocks for the session-aware search pipeline."""

from .context import RetrievalContext
from .context import build_retrieval_context

__all__ = ["RetrievalContext", "build_retrieval_context"]
