"""Factory that assembles the reranker wrapper chain from environment variables.

Chain (outermost → innermost):
    TwoStageReranker → CachedReranker → AsyncReranker → Reranker
Each layer is optional; unset env vars leave the chain unchanged.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.internal.retrieval.reranker import Reranker


def _flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def _wrap_async(base: "Reranker") -> "Reranker":
    """Wrap *base* in AsyncReranker → CachedReranker when RERANKER_ASYNC is set."""
    if not _flag("RERANKER_ASYNC"):
        return base
    from src.internal.retrieval.async_reranker import AsyncReranker
    from src.internal.retrieval.cached_reranker import CachedReranker

    return CachedReranker.from_env(AsyncReranker.from_env(base))


def build_reranker_from_env() -> "Reranker | None":
    """Return the fully-composed reranker chain, or None when disabled."""
    from src.internal.retrieval.reranker import Reranker, RerankerConfig

    heavy = Reranker.from_env()
    if heavy is None:
        return None

    if not _flag("RERANKER_TWO_STAGE"):
        return _wrap_async(heavy)

    from src.internal.retrieval.two_stage_reranker import TwoStageReranker

    fast_model = os.environ.get("RERANKER_FAST_MODEL")
    if fast_model:
        fast_cfg = RerankerConfig(
            provider=os.environ.get("RERANKER_PROVIDER", "local"),  # type: ignore[arg-type]
            model=fast_model,
            batch_size=int(os.environ.get("RERANKER_BATCH_SIZE", "32")),
            device=os.environ.get("RERANKER_DEVICE", "cpu"),
            api_key=os.environ.get("COHERE_API_KEY"),
        )
        fast = Reranker(fast_cfg)
    else:
        fast = heavy

    return TwoStageReranker.from_env(_wrap_async(fast), _wrap_async(heavy))
