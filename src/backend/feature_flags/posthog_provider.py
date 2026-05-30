"""Optional PostHog-backed feature flag provider."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PostHogFeatureFlagProvider:
    """Adapter for a PostHog-like client.

    The client is injected to keep this module dependency-light. Any object with
    ``feature_enabled(flag_key, distinct_id, person_properties=...)`` is enough.
    """

    client: Any | None = None

    def feature_enabled(
        self,
        flag_key: str,
        user_id: str | None = None,
        user_properties: dict[str, Any] | None = None,
        *,
        default: bool = False,
    ) -> bool:
        if self.client is None or user_id is None:
            return default
        try:
            set_properties = getattr(self.client, "set", None)
            if callable(set_properties):
                set_properties(distinct_id=user_id, properties=user_properties)
            enabled = self.client.feature_enabled(
                flag_key,
                str(user_id),
                person_properties=user_properties,
            )
            return default if enabled is None else bool(enabled)
        except Exception as exc:
            logger.warning(
                "Feature flag lookup failed for %s and user %s: %s",
                flag_key,
                user_id,
                exc,
            )
            return default


__all__ = ["PostHogFeatureFlagProvider"]
