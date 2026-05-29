"""Pydantic models for tenant and invitation management.

The sampled Onyx models imported CustomerTier (EE) and ApplicationStatus
(Onyx settings) — both replaced with repo equivalents from src/utils/license.py
and src/configs/. Cloud-only models (Stripe checkout, billing information,
schema management) are kept as stubs so callers that reference them compile,
but the corresponding endpoints return 501.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from src.configs import Tier
from src.utils.license import ApplicationStatus


class CreateTenantRequest(BaseModel):
    tenant_id: str
    initial_admin_email: str


class ProductGatingRequest(BaseModel):
    tenant_id: str
    application_status: ApplicationStatus


class ProductGatingResponse(BaseModel):
    updated: bool
    error: str | None = None


class TierUpdateRequest(BaseModel):
    tenant_id: str
    customer_tier: Tier
    trial_end: datetime | None = None


class TierUpdateResponse(BaseModel):
    updated: bool
    error: str | None = None


class AnonymousUserPath(BaseModel):
    anonymous_user_path: str | None = None


class RequestInviteRequest(BaseModel):
    email: str


class RequestInviteResponse(BaseModel):
    success: bool
    message: str


class PendingUserSnapshot(BaseModel):
    email: str


class ApproveUserRequest(BaseModel):
    email: str


# Cloud-only stubs — kept so imports compile; not used by any live endpoint.
class BillingInformation(BaseModel):
    stripe_subscription_id: str = ""
    status: str = "unknown"
    current_period_start: datetime = datetime.min
    current_period_end: datetime = datetime.min
    number_of_seats: int = 0
    cancel_at_period_end: bool = False
    canceled_at: datetime | None = None
    trial_start: datetime | None = None
    trial_end: datetime | None = None
    seats: int = 0
    payment_method_enabled: bool = False
    customer_tier: Tier | None = None


class CreateCheckoutSessionRequest(BaseModel):
    billing_period: Literal["monthly", "annual"] = "monthly"
    seats: int | None = None
    email: str | None = None


class CheckoutSessionCreationResponse(BaseModel):
    id: str


class SubscriptionSessionResponse(BaseModel):
    sessionId: str


__all__ = [
    "AnonymousUserPath",
    "ApproveUserRequest",
    "BillingInformation",
    "CheckoutSessionCreationResponse",
    "CreateCheckoutSessionRequest",
    "CreateTenantRequest",
    "PendingUserSnapshot",
    "ProductGatingRequest",
    "ProductGatingResponse",
    "RequestInviteRequest",
    "RequestInviteResponse",
    "SubscriptionSessionResponse",
    "TierUpdateRequest",
    "TierUpdateResponse",
]
