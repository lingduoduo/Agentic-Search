"""Settings API.

py.
Redis, SQLAlchemy, and the license cache are replaced with the
project's file-backed license store (src/utils/license.py).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from src.backend.configs import AppSettings
from src.backend.configs import Tier
from src.backend.utils.license import ApplicationStatus
from src.backend.utils.license import get_license_status
from src.backend.utils.license import load_stored_license
from src.backend.utils.license import verify_license_signature
from src.backend.utils.license_expiry import get_grace_period_end
from src.backend.utils.tier import get_tier

logger = logging.getLogger(__name__)


class Settings(BaseModel):
    """Application settings as exposed to the UI."""

    application_status: ApplicationStatus | None = None
    ee_features_enabled: bool = False
    tier: str = Tier.FREE.value
    license_enforcement_enabled: bool = False


def _load_license_status(
    app_settings: AppSettings,
) -> tuple[ApplicationStatus | None, bool, str]:
    """Return (application_status, ee_features_enabled, tier_value)."""
    if not app_settings.license_enforcement_enabled:
        current_tier = get_tier(app_settings)
        ee_on = current_tier == Tier.ENTERPRISE
        return None, ee_on, current_tier.value

    license_data = load_stored_license()
    if not license_data:
        return ApplicationStatus.GATED_ACCESS, False, Tier.FREE.value

    try:
        payload = verify_license_signature(license_data)
    except (ValueError, Exception):
        return ApplicationStatus.GATED_ACCESS, False, Tier.FREE.value

    grace_end = get_grace_period_end(payload.expires_at)
    status = get_license_status(payload, grace_end)
    ee_on = status != ApplicationStatus.GATED_ACCESS
    tier_val = Tier.ENTERPRISE.value if ee_on else Tier.FREE.value
    return status, ee_on, tier_val


def create_settings_router(app_settings: AppSettings) -> APIRouter:
    """Return a router exposing the /settings endpoint."""

    router = APIRouter(prefix="/settings", tags=["settings"])

    @router.get("")
    def get_settings() -> Settings:
        """Return current application status and tier for the UI."""
        status, ee_on, tier_val = _load_license_status(app_settings)
        return Settings(
            application_status=status,
            ee_features_enabled=ee_on,
            tier=tier_val,
            license_enforcement_enabled=app_settings.license_enforcement_enabled,
        )

    return router


__all__ = ["Settings", "create_settings_router"]
