"""Auth endpoint rate limiting — in-memory sliding window, no Redis required.

Replaces the fastapi-limiter/Redis version that depended on onyx packages.
Configuration is read from environment variables; defaults match the original
onyx values (100 requests per 60 seconds).
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from typing import List

from fastapi import Depends, HTTPException, Request

_AUTH_RATE_LIMITING_ENABLED: bool = os.environ.get(
    "AUTH_RATE_LIMITING_ENABLED", ""
).lower() in ("1", "true", "yes")
_RATE_LIMIT_MAX_REQUESTS: int = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "100"))
_RATE_LIMIT_WINDOW_SECONDS: int = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))

_lock = threading.Lock()
_windows: dict[str, list[float]] = defaultdict(list)


async def setup_auth_limiter() -> None:
    """No-op — in-memory limiter requires no setup."""


async def close_auth_limiter() -> None:
    """No-op — in-memory limiter holds no external connections."""


def _make_key(request: Request) -> str:
    ip_part = request.client.host if request.client else "unknown"
    ua_part = request.headers.get("user-agent", "none").replace(" ", "_")
    return f"{ip_part}-{ua_part}"


def _make_limiter(max_requests: int, window_seconds: int) -> Callable:
    async def limiter(request: Request) -> None:
        key = _make_key(request)
        now = time.monotonic()
        cutoff = now - window_seconds

        with _lock:
            active = [t for t in _windows[key] if t > cutoff]
            if len(active) >= max_requests:
                _windows[key] = active
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests. Please try again later.",
                )
            active.append(now)
            _windows[key] = active

    return limiter


def get_auth_rate_limiters() -> List[Callable]:
    if not _AUTH_RATE_LIMITING_ENABLED:
        return []

    return [
        Depends(_make_limiter(_RATE_LIMIT_MAX_REQUESTS, _RATE_LIMIT_WINDOW_SECONDS))
    ]
