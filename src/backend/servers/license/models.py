"""Pydantic models for the license API.

The sampled Onyx models imported ExpiryWarningStage from ee.onyx and
ApplicationStatus from onyx.server.settings.models. Both are replaced
with repo equivalents from src/utils/.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from src.backend.utils.license import ApplicationStatus
from src.backend.utils.license_expiry import ExpiryWarningStage


class PlanType(str, Enum):
    MONTHLY = "monthly"
    ANNUAL = "annual"


class LicenseSource(str, Enum):
    AUTO_FETCH = "auto_fetch"
    MANUAL_UPLOAD = "manual_upload"


class LicenseStatusResponse(BaseModel):
    has_license: bool
    max_seats: int | None = None
    tenant_id: str | None = None
    expires_at: datetime | None = None
    grace_period_end: datetime | None = None
    status: ApplicationStatus | None = None
    expiry_warning_stage: ExpiryWarningStage = ExpiryWarningStage.NONE
    source: LicenseSource | None = None
    features: list[str] = []


class LicenseResponse(BaseModel):
    success: bool
    message: str | None = None
    expires_at: datetime | None = None
    max_seats: int | None = None
    tenant_id: str | None = None


class LicenseUploadResponse(BaseModel):
    success: bool
    message: str | None = None


class SeatUsageResponse(BaseModel):
    total_seats: int
    used_seats: int
    available_seats: int


__all__ = [
    "LicenseResponse",
    "LicenseSource",
    "LicenseStatusResponse",
    "LicenseUploadResponse",
    "PlanType",
    "SeatUsageResponse",
]
