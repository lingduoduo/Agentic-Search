"""Enterprise settings API router.

Provides branding/UI configuration endpoints. The previous version
depended on get_kv_store(), get_default_file_store(), ScimDAL, Meechum
OAuth token refresh, MULTI_TENANT context vars, and AppError error codes.

This implementation uses:
  - File-backed store (store.py in this package)
  - src/utils/tier.py for Enterprise tier gating
  - Standard FastAPI HTTPException for errors
  - AppSettings.auth.super_users for admin auth

SCIM token management and OAuth token refresh are EE/cloud features with
no equivalent in this repo; those endpoints return HTTP 501.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Response
from fastapi import UploadFile

from src.backend.auth import AuthenticatedUser
from src.backend.configs import AppSettings
from src.backend.configs import Tier
from src.backend.servers.enterprise_settings.models import AnalyticsScriptUpload
from src.backend.servers.enterprise_settings.models import EnterpriseSettings
from src.backend.servers.enterprise_settings.store import get_logo_bytes
from src.backend.servers.enterprise_settings.store import load_analytics_script
from src.backend.servers.enterprise_settings.store import load_settings
from src.backend.servers.enterprise_settings.store import store_analytics_script
from src.backend.servers.enterprise_settings.store import store_settings
from src.backend.servers.enterprise_settings.store import upload_logo
from src.backend.utils.tier import get_tier
from src.backend.utils.tier import tier_at_least
from src.backend.servers._auth import make_require_admin

logger = logging.getLogger(__name__)


def create_enterprise_settings_routers(
    app_settings: AppSettings,
) -> tuple[APIRouter, APIRouter]:
    """Return ``(admin_router, basic_router)`` for enterprise settings endpoints."""

    admin_router = APIRouter(
        prefix="/admin/enterprise-settings", tags=["enterprise-settings"]
    )
    basic_router = APIRouter(
        prefix="/enterprise-settings", tags=["enterprise-settings"]
    )

    _require_admin = make_require_admin(app_settings)

    # ---------------------------------------------------------------------------
    # Settings read / write
    # ---------------------------------------------------------------------------

    @basic_router.get("")
    def fetch_settings() -> EnterpriseSettings:
        """Return current enterprise settings (no auth required)."""
        return load_settings()

    @admin_router.put("")
    def put_settings(
        settings: EnterpriseSettings,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> None:
        """Update enterprise settings. Enterprise-tier fields are gated."""
        current_tier = get_tier(app_settings)
        if not tier_at_least(current_tier, Tier.ENTERPRISE):
            existing = load_settings()
            if settings.custom_help_link_url != existing.custom_help_link_url or (
                settings.custom_help_link_label != existing.custom_help_link_label
            ):
                raise HTTPException(
                    status_code=402,
                    detail="Custom help link requires the Enterprise plan.",
                )
            if settings.hide_branding != existing.hide_branding:
                raise HTTPException(
                    status_code=402,
                    detail="Hiding branding requires the Enterprise plan.",
                )
        store_settings(settings)

    # ---------------------------------------------------------------------------
    # Logo upload / fetch
    # ---------------------------------------------------------------------------

    @admin_router.put("/logo")
    def put_logo(
        file: UploadFile,
        is_logotype: bool = False,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> None:
        """Upload a custom logo or logotype (PNG / JPEG only)."""
        upload_logo(file, is_logotype=is_logotype)

    @basic_router.get("/logo")
    def fetch_logo(is_logotype: bool = False) -> Response:
        """Return the stored logo binary."""
        result = get_logo_bytes(is_logotype=is_logotype)
        if result is None:
            raise HTTPException(status_code=404, detail="No logo found.")
        raw, mime = result
        return Response(
            content=raw, media_type=mime, headers={"Cache-Control": "no-cache"}
        )

    @basic_router.get("/logotype")
    def fetch_logotype() -> Response:
        """Return the stored logotype binary."""
        result = get_logo_bytes(is_logotype=True)
        if result is None:
            raise HTTPException(status_code=404, detail="No logotype found.")
        raw, mime = result
        return Response(content=raw, media_type=mime)

    # ---------------------------------------------------------------------------
    # Analytics script
    # ---------------------------------------------------------------------------

    @admin_router.put("/custom-analytics-script")
    def upload_custom_analytics_script(
        script_upload: AnalyticsScriptUpload,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> None:
        """Store a custom analytics script (requires CUSTOM_ANALYTICS_SECRET_KEY env var)."""
        try:
            store_analytics_script(script_upload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @basic_router.get("/custom-analytics-script")
    def fetch_custom_analytics_script() -> str | None:
        """Return the stored analytics script, or null if none."""
        return load_analytics_script()

    # ---------------------------------------------------------------------------
    # Stubs for EE/cloud-only features
    # ---------------------------------------------------------------------------

    @admin_router.get("/scim/token")
    def get_scim_token(_: AuthenticatedUser = Depends(_require_admin)) -> None:
        """SCIM token management is not available in this deployment."""
        raise HTTPException(status_code=501, detail="SCIM is not available.")

    @admin_router.post("/scim/token", status_code=201)
    def create_scim_token(_: AuthenticatedUser = Depends(_require_admin)) -> None:
        """SCIM token management is not available in this deployment."""
        raise HTTPException(status_code=501, detail="SCIM is not available.")

    @basic_router.post("/refresh-token")
    def refresh_token() -> None:
        """OAuth token refresh is not available in this deployment."""
        raise HTTPException(
            status_code=501, detail="OAuth token refresh is not available."
        )

    return admin_router, basic_router


__all__ = ["create_enterprise_settings_routers"]
