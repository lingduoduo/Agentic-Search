"""Connector sync API router.

The cc_pair.py triggered Celery tasks via Redis for permission
sync and external group sync — all of which require the full
infrastructure (SQLAlchemy, Celery, Redis, multi-tenant context vars).

This repo uses AgenticSearchStore and stores sync state as IndexAttemptRecords.
The endpoints are adapted accordingly:

  GET  /manage/admin/connector/{connector_id}/last-sync
       Returns the created_at timestamp of the most recent index attempt.

  POST /manage/admin/connector/{connector_id}/sync
       Creates a new index attempt (status "not_started") for the connector.

  GET  /manage/admin/connector/{connector_id}/last-group-sync
  POST /manage/admin/connector/{connector_id}/sync-groups
       External group sync has no equivalent in this self-hosted repo;
       these endpoints return 501.

Admin access requires the caller's user ID or email to be listed in
AppSettings.auth.super_users.
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from pydantic import BaseModel

from src.backend.auth import AuthenticatedUser
from src.backend.configs import AppSettings
from src.backend.db import AgenticSearchStore
from src.backend.servers._auth import make_require_admin

logger = logging.getLogger(__name__)


class StatusResponse(BaseModel):
    success: bool
    message: str


def _require_connector(store: AgenticSearchStore, connector_id: str) -> None:
    if store.get_connector(connector_id) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Connector {connector_id!r} not found.",
        )


def _get_latest_sync_at(
    store: AgenticSearchStore,
    connector_id: str,
) -> datetime | None:
    attempts = store.list_index_attempts(connector_id=connector_id)
    timestamps = [
        datetime.fromisoformat(attempt.created_at)
        for attempt in attempts
        if attempt.created_at
    ]
    return max(timestamps, default=None)


def create_documents_router(
    store: AgenticSearchStore,
    app_settings: AppSettings,
) -> APIRouter:
    """Return an APIRouter for connector-management endpoints."""

    router = APIRouter(prefix="/manage", tags=["documents"])

    _require_admin = make_require_admin(app_settings)

    @router.get("/admin/connector/{connector_id}/last-sync")
    def get_connector_last_sync(
        connector_id: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> datetime | None:
        """Return the timestamp of the most recent index attempt, or None."""
        _require_connector(store, connector_id)
        return _get_latest_sync_at(store, connector_id)

    @router.post("/admin/connector/{connector_id}/sync")
    def trigger_connector_sync(
        connector_id: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> StatusResponse:
        """Queue a new index attempt for the connector.

        The actual indexing work is performed by a background process that
        polls for ``not_started`` attempts. This endpoint only enqueues the job.
        """
        _require_connector(store, connector_id)
        attempt = store.create_index_attempt(connector_id=connector_id)
        logger.info(
            "Sync attempt created: connector=%s attempt=%s", connector_id, attempt.id
        )
        return StatusResponse(
            success=True,
            message=f"Sync queued (attempt {attempt.id}).",
        )

    @router.get("/admin/connector/{connector_id}/last-group-sync")
    def get_connector_last_group_sync(
        connector_id: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> None:
        """External group sync is not supported in this single-tenant deployment."""
        _require_connector(store, connector_id)
        raise HTTPException(
            status_code=501,
            detail="External group sync is not available in this deployment.",
        )

    @router.post("/admin/connector/{connector_id}/sync-groups")
    def trigger_connector_group_sync(
        connector_id: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> None:
        """External group sync is not supported in this single-tenant deployment."""
        _require_connector(store, connector_id)
        raise HTTPException(
            status_code=501,
            detail="External group sync is not available in this deployment.",
        )

    return router


__all__ = ["StatusResponse", "create_documents_router"]
