"""PostHog client initialisation and helper utilities.

The ``posthog`` package is optional — all functions no-op gracefully when it
is not installed or when ``TelemetrySettings.posthog_api_key`` is unset.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import unquote

logger = logging.getLogger(__name__)

try:
    from posthog import Posthog as _Posthog

    _POSTHOG_AVAILABLE = True
except ImportError:
    _Posthog = None  # type: ignore[assignment,misc]
    _POSTHOG_AVAILABLE = False


def _on_error(error: Any, items: Any) -> None:
    logger.error("PostHog delivery error: %s (items: %s)", error, items)


def build_posthog_client(
    api_key: str | None,
    host: str = "https://us.i.posthog.com",
    *,
    debug: bool = False,
) -> Any | None:
    """Return a configured ``Posthog`` instance, or ``None`` if unavailable."""
    if not api_key or not _POSTHOG_AVAILABLE:
        return None
    return _Posthog(
        project_api_key=api_key,
        host=host,
        debug=debug,
        on_error=_on_error,
    )


def alias_user(client: Any | None, distinct_id: str, anonymous_id: str) -> None:
    """Link an anonymous distinct_id to an identified user in PostHog.

    No-ops when the IDs are identical (returning user whose PostHog cookie
    already contains their identified ID) or when no client is available.
    """
    if not client or anonymous_id == distinct_id:
        return
    try:
        client.alias(previous_id=anonymous_id, distinct_id=distinct_id)
        client.flush()
    except Exception as exc:
        logger.error("PostHog alias failed: %s", exc)


def parse_posthog_cookie(cookie_value: str) -> dict[str, Any] | None:
    """Parse a URL-encoded JSON PostHog cookie.

    Returns a dict containing at least ``distinct_id``, or ``None`` if the
    cookie is malformed or missing ``distinct_id``.
    """
    try:
        data = json.loads(unquote(cookie_value))
        distinct_id = data.get("distinct_id")
        if not distinct_id or not isinstance(distinct_id, str):
            return None
        return data
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
        logger.warning("Failed to parse PostHog cookie: %s", exc)
        return None


def get_anon_id_from_request(
    request: Any,
    api_key: str | None,
) -> str | None:
    """Extract the anonymous ``distinct_id`` from the PostHog cookie on *request*."""
    if not api_key:
        return None
    cookie_name = f"ph_{api_key}_posthog"
    cookie_value = getattr(request, "cookies", {}).get(cookie_name)
    if not cookie_value:
        return None
    parsed = parse_posthog_cookie(cookie_value)
    return parsed.get("distinct_id") if parsed else None


__all__ = [
    "alias_user",
    "build_posthog_client",
    "get_anon_id_from_request",
    "parse_posthog_cookie",
]
