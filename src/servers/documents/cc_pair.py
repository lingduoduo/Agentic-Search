"""Connector sync API router.

The sampled Onyx cc_pair.py triggered Celery tasks via Redis for permission
sync and external group sync — all of which require the full Onyx
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
from fastapi import Request
from pydantic import BaseModel

from src.auth import AuthenticatedUser
from src.auth import user_from_headers
from src.configs import AppSettings
from src.db import AgenticSearchStore

logger = logging.getLogger(__name__)


class StatusResponse(BaseModel):
    success: bool
    message: str


def create_documents_router(
    store: AgenticSearchStore,
    app_settings: AppSettings,
) -> APIRouter:
    """Return an APIRouter for connector-management endpoints."""

    router = APIRouter(prefix="/manage", tags=["documents"])

    def _require_admin(request: Request) -> AuthenticatedUser:
        user = user_from_headers(request.headers)
        if user is None or user.is_anonymous:
            raise HTTPException(status_code=401, detail="Authentication required.")
        super_users = app_settings.auth.super_users
        if user.id not in super_users and (
            user.email is None or user.email not in super_users
        ):
            raise HTTPException(status_code=403, detail="Admin access required.")
        return user

    def _get_connector_or_404(connector_id: str) -> None:
        if store.get_connector(connector_id) is None:
            raise HTTPException(
                status_code=404,
                detail=f"Connector {connector_id!r} not found.",
            )

    @router.get("/admin/connector/{connector_id}/last-sync")
    def get_connector_last_sync(
        connector_id: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> datetime | None:
        """Return the timestamp of the most recent index attempt, or None."""
        _get_connector_or_404(connector_id)
        attempts = store.list_index_attempts(connector_id=connector_id)
        if not attempts:
            return None
        latest = max(attempts, key=lambda a: a.created_at or "")
        if not latest.created_at:
            return None
        return datetime.fromisoformat(latest.created_at)

    @router.post("/admin/connector/{connector_id}/sync")
    def trigger_connector_sync(
        connector_id: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> StatusResponse:
        """Queue a new index attempt for the connector.

        The actual indexing work is performed by a background process that
        polls for ``not_started`` attempts. This endpoint only enqueues the job.
        """
        _get_connector_or_404(connector_id)
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
        raise HTTPException(
            status_code=501,
            detail="External group sync is not available in this deployment.",
        )

    return router


__all__ = ["StatusResponse", "create_documents_router"]
