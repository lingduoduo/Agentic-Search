import logging
import os
import uuid
from typing import Any

from sentry_sdk.types import Event

logger = logging.getLogger(__name__)

_instance_id_resolved = False


def _get_or_generate_uuid() -> str:
    """Return a stable instance UUID from env or generate a random one."""
    stored = os.environ.get("INSTANCE_UUID")
    if stored:
        return stored
    return str(uuid.uuid4())


def _add_instance_tags(
    event: Event,
    hint: dict[str, Any],  # noqa: ARG001
) -> Event | None:
    """Sentry before_send hook that lazily attaches instance identification tags."""
    global _instance_id_resolved

    if _instance_id_resolved:
        return event

    try:
        import sentry_sdk

        from src.shared_configs.configs import MULTI_TENANT

        if MULTI_TENANT:
            instance_id = "multi-tenant-cloud"
        else:
            instance_id = _get_or_generate_uuid()

        sentry_sdk.set_tag("instance_id", instance_id)
        event.setdefault("tags", {})["instance_id"] = instance_id
        _instance_id_resolved = True
    except Exception:
        logger.debug("Failed to resolve instance_id for Sentry tagging")

    return event
