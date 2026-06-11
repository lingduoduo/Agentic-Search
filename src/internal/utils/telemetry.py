"""PostHog event capture and user identification helpers.

Callers pass an explicit PostHog client (obtained from
``posthog_client.build_posthog_client``) so this module has no global
side-effects at import time and is trivially testable.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def event_telemetry(
    client: Any | None,
    distinct_id: str,
    event: str,
    properties: dict[str, Any] | None = None,
) -> None:
    """Capture *event* for *distinct_id* in PostHog, flushing immediately.

    No-ops when *client* is ``None`` (PostHog not configured).
    Properties are logged before capture so real client IPs (PII) never
    reach the application log aggregator.
    """
    if not client:
        return
    logger.info("Capturing PostHog event: %s %s %s", distinct_id, event, properties)
    try:
        client.capture(distinct_id, event, properties)
        client.flush()
    except Exception as exc:
        logger.error("PostHog capture failed: %s", exc)


def identify_user(
    client: Any | None,
    distinct_id: str,
    properties: dict[str, Any] | None = None,
) -> None:
    """Create or update a PostHog person profile, flushing immediately."""
    if not client:
        return
    try:
        client.identify(distinct_id, properties)
        client.flush()
    except Exception as exc:
        logger.error("PostHog identify failed: %s", exc)


__all__ = [
    "event_telemetry",
    "identify_user",
]
