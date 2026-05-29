"""In-process message rate limiting for the query API.

The sampled Onyx version used SQLAlchemy ORM queries over a `token_count`
field in `ChatMessage`, user-group rate-limit tables, and global limits stored
in the DB. This repo has no token-count field and no rate-limit tables.

This module implements a lightweight in-memory sliding-window counter instead:
one window per user_id, purged on each check. It works correctly for a
single-process deployment and is a no-op (always allows) if the limit is
set to 0.

Thread-safety: uses a simple lock since FastAPI runs handlers in a thread pool.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

from fastapi import HTTPException

_lock = threading.Lock()
_windows: dict[str, list[float]] = defaultdict(list)

_DEFAULT_MAX_MESSAGES = 100
_DEFAULT_WINDOW_SECONDS = 3600  # 1 hour


def _get_cutoff_time(window_seconds: float) -> float:
    return time.monotonic() - window_seconds


def _is_rate_limited(timestamps: list[float], cutoff: float, max_count: int) -> bool:
    return sum(1 for t in timestamps if t > cutoff) >= max_count


def check_message_rate_limit(
    user_id: str,
    max_messages: int = _DEFAULT_MAX_MESSAGES,
    window_seconds: float = _DEFAULT_WINDOW_SECONDS,
) -> None:
    """Raise HTTP 429 if *user_id* has exceeded *max_messages* in *window_seconds*.

    Passing ``max_messages=0`` disables the limit entirely.
    """
    if max_messages <= 0:
        return

    now = time.monotonic()
    cutoff = _get_cutoff_time(window_seconds)

    with _lock:
        timestamps = _windows[user_id]
        # Prune expired entries to bound memory growth.
        active = [t for t in timestamps if t > cutoff]
        if _is_rate_limited(active, cutoff, max_messages):
            _windows[user_id] = active
            raise HTTPException(
                status_code=429,
                detail="Message rate limit exceeded. Try again later.",
            )
        active.append(now)
        _windows[user_id] = active


def reset_rate_limit(user_id: str) -> None:
    """Clear the rate-limit window for *user_id*. Useful in tests."""
    with _lock:
        _windows.pop(user_id, None)


__all__ = [
    "check_message_rate_limit",
    "reset_rate_limit",
]
