"""Environment-driven DB seeding for Agentic Search.

On startup, ``seed_db(store)`` reads ``ENV_SEED_CONFIGURATION`` (a JSON string)
and upserts the declared users and connectors into ``AgenticSearchStore``.

This is the repo-native port of the seeding module. repo-specific
concepts (personas, LLM providers, tools, enterprise settings, logos, analytics
scripts) are omitted because they have no equivalent in this repo's data model.

Example env var:
    ENV_SEED_CONFIGURATION='{"users":[{"id":"u1","email":"admin@example.com"}]}'
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pydantic import BaseModel

from src.backend.db import AgenticSearchStore
from src.backend.db.models import ConnectorConfig
from src.backend.db.models import UserRecord

logger = logging.getLogger(__name__)

_SEED_CONFIG_ENV_VAR = "ENV_SEED_CONFIGURATION"


# ---------------------------------------------------------------------------
# Seed models
# ---------------------------------------------------------------------------


class UserSeed(BaseModel):
    """Fields forwarded to ``AgenticSearchStore.upsert_user``."""

    id: str
    email: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = {}


class ConnectorSeed(BaseModel):
    """Fields forwarded to ``AgenticSearchStore.upsert_connector``."""

    id: str
    name: str
    source: str
    config: dict[str, Any] = {}
    enabled: bool = True
    metadata: dict[str, Any] = {}


class SeedConfiguration(BaseModel):
    """Top-level seed configuration loaded from ``ENV_SEED_CONFIGURATION``."""

    users: list[UserSeed] = []
    connectors: list[ConnectorSeed] = []
    # IDs to surface as admin in AppSettings.auth.super_users.
    # Recorded here for documentation; no DB action is taken.
    admin_user_ids: list[str] = []


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def get_seed_config() -> SeedConfiguration | None:
    raw = os.getenv(_SEED_CONFIG_ENV_VAR)
    if not raw:
        return None
    try:
        return SeedConfiguration.model_validate_json(raw)
    except Exception as exc:
        logger.error("Failed to parse %s: %s", _SEED_CONFIG_ENV_VAR, exc)
        return None


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------


def _seed_users(store: AgenticSearchStore, users: list[UserSeed]) -> None:
    if not users:
        return
    logger.info("Seeding %d user(s).", len(users))
    for u in users:
        try:
            store.upsert_user(
                UserRecord(
                    id=u.id,
                    email=u.email,
                    name=u.name,
                    metadata=u.metadata,
                )
            )
            logger.debug("Seeded user %s (%s).", u.id, u.email)
        except Exception as exc:
            logger.error("Failed to seed user %s: %s", u.id, exc)


def _seed_connectors(
    store: AgenticSearchStore, connectors: list[ConnectorSeed]
) -> None:
    if not connectors:
        return
    logger.info("Seeding %d connector(s).", len(connectors))
    for c in connectors:
        try:
            store.upsert_connector(
                ConnectorConfig(
                    id=c.id,
                    name=c.name,
                    source=c.source,
                    config=c.config,
                    enabled=c.enabled,
                    metadata=c.metadata,
                )
            )
            logger.debug("Seeded connector %s (%s).", c.id, c.name)
        except Exception as exc:
            logger.error("Failed to seed connector %s: %s", c.id, exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def seed_db(store: AgenticSearchStore) -> None:
    """Seed *store* from ``ENV_SEED_CONFIGURATION``. No-op if not set."""
    config = get_seed_config()
    if config is None:
        logger.debug("No seed configuration found (%s not set).", _SEED_CONFIG_ENV_VAR)
        return

    _seed_users(store, config.users)
    _seed_connectors(store, config.connectors)

    if config.admin_user_ids:
        logger.info(
            "Seed config declares %d admin user ID(s): %s. "
            "Add these to AppSettings.auth.super_users to grant admin access.",
            len(config.admin_user_ids),
            config.admin_user_ids,
        )

    logger.info("Seeding complete.")


__all__ = [
    "ConnectorSeed",
    "SeedConfiguration",
    "UserSeed",
    "get_seed_config",
    "seed_db",
]
