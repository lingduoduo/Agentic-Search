"""Billing admin API.

Adapted from the sampled Onyx ee/onyx/server/billing/api.py.
Changes from the original:
- Redis circuit breaker replaced with a thread-safe in-memory flag.
- MULTI_TENANT removed — single-tenant self-hosted deployment only.
- OnyxError / OnyxErrorCode replaced with HTTPException.
- License data read from the file-backed store (same path as license/api.py).
- Stripe key and web domain read from AppSettings instead of module-level env vars.
- require_permission / User replaced with the project's _require_admin pattern.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path

import httpx
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from pydantic import BaseModel

from src.auth import AuthenticatedUser
from src.auth import user_from_headers
from src.configs import AppSettings
from src.servers.billing.models import BillingInformationResponse
from src.servers.billing.models import CreateCheckoutSessionRequest
from src.servers.billing.models import CreateCheckoutSessionResponse
from src.servers.billing.models import CreateCustomerPortalSessionRequest
from src.servers.billing.models import CreateCustomerPortalSessionResponse
from src.servers.billing.models import EndTrialResponse
from src.servers.billing.models import SeatUpdateRequest
from src.servers.billing.models import SeatUpdateResponse
from src.servers.billing.models import StripePublishableKeyResponse
from src.servers.billing.models import SubscriptionStatusResponse
from src.servers.billing.service import create_checkout_session as _create_checkout
from src.servers.billing.service import (
    create_customer_portal_session as _create_portal,
)
from src.servers.billing.service import end_trial as _end_trial
from src.servers.billing.service import get_billing_information as _get_billing
from src.servers.billing.service import update_seat_count as _update_seats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory circuit breaker (replaces Redis from sampled code)
# ---------------------------------------------------------------------------

_circuit_lock = threading.Lock()
_circuit_open: bool = False


def _is_billing_circuit_open() -> bool:
    with _circuit_lock:
        return _circuit_open


def _open_billing_circuit() -> None:
    global _circuit_open
    with _circuit_lock:
        _circuit_open = True
    logger.warning("Billing circuit breaker opened. Stripe requests disabled.")


def _close_billing_circuit() -> None:
    global _circuit_open
    with _circuit_lock:
        _circuit_open = False
    logger.info("Billing circuit breaker closed. Stripe requests re-enabled.")


# ---------------------------------------------------------------------------
# License file helpers (mirrors logic in license/api.py)
# ---------------------------------------------------------------------------


def _license_dat_path() -> Path:
    data_dir = Path(
        os.environ.get(
            "AGENTIC_SEARCH_DATA_DIR",
            Path.home() / ".local/share/agentic_search",
        )
    )
    return data_dir / "license.dat"


def _load_license_data() -> str | None:
    path = _license_dat_path()
    return path.read_text().strip() if path.exists() else None


# ---------------------------------------------------------------------------
# Stripe publishable key cache
# ---------------------------------------------------------------------------

_stripe_key_cache: str | None = None
_stripe_key_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_billing_router(app_settings: AppSettings) -> APIRouter:
    """Return an APIRouter with billing admin endpoints."""

    router = APIRouter(prefix="/admin/billing", tags=["billing"])

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

    def _require_proxy_url() -> str:
        """Return the billing proxy base URL or raise 501."""
        base = app_settings.cloud_data_plane_url
        if not base:
            raise HTTPException(
                status_code=501,
                detail=(
                    "Billing is not configured. "
                    "Set AGENTIC_SEARCH_CLOUD_DATA_PLANE_URL to enable."
                ),
            )
        return f"{base.rstrip('/')}/proxy"

    @router.post("/create-checkout-session")
    async def create_checkout_session(
        request: CreateCheckoutSessionRequest | None = None,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> CreateCheckoutSessionResponse:
        """Create a Stripe checkout session for a new subscription or renewal."""
        proxy_url = _require_proxy_url()
        license_data = _load_license_data()
        billing_period = request.billing_period if request else "monthly"
        seats = request.seats if request else None
        email = request.email if request else None
        redirect_url = f"{app_settings.web_domain}/admin/billing?checkout=success"

        return await _create_checkout(
            base_url=proxy_url,
            billing_period=billing_period,
            seats=seats,
            email=email,
            license_data=license_data,
            redirect_url=redirect_url,
        )

    @router.post("/create-customer-portal-session")
    async def create_customer_portal_session(
        request: CreateCustomerPortalSessionRequest | None = None,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> CreateCustomerPortalSessionResponse:
        """Create a Stripe customer portal session."""
        proxy_url = _require_proxy_url()
        license_data = _load_license_data()
        if not license_data:
            raise HTTPException(status_code=400, detail="No license found.")
        return_url = (
            request.return_url if request else None
        ) or f"{app_settings.web_domain}/admin/billing"
        flow_type = request.flow_type if request else None
        return await _create_portal(
            base_url=proxy_url,
            license_data=license_data,
            return_url=return_url,
            flow_type=flow_type,
        )

    @router.get("/billing-information")
    async def get_billing_information(
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> BillingInformationResponse | SubscriptionStatusResponse:
        """Return subscription status and billing details."""
        license_data = _load_license_data()
        if not license_data:
            return SubscriptionStatusResponse(subscribed=False)

        if _is_billing_circuit_open():
            raise HTTPException(
                status_code=503,
                detail="Stripe connection temporarily disabled. Click 'Connect to Stripe' to retry.",
            )

        proxy_url = _require_proxy_url()
        try:
            return await _get_billing(base_url=proxy_url, license_data=license_data)
        except HTTPException as e:
            if e.status_code in (502, 503, 504):
                _open_billing_circuit()
            raise

    @router.post("/seats/update")
    async def update_seats(
        request: SeatUpdateRequest,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> SeatUpdateResponse:
        """Update the seat count for the current subscription."""
        license_data = _load_license_data()
        if not license_data:
            raise HTTPException(status_code=400, detail="No license found.")
        proxy_url = _require_proxy_url()
        return await _update_seats(
            base_url=proxy_url,
            new_seat_count=request.new_seat_count,
            license_data=license_data,
        )

    @router.post("/end-trial")
    async def end_trial(
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> EndTrialResponse:
        """End the current trial — cloud deployments only."""
        return await _end_trial()

    @router.get("/stripe-publishable-key")
    async def get_stripe_publishable_key() -> StripePublishableKeyResponse:
        """Return the Stripe publishable key (cached after first fetch)."""
        global _stripe_key_cache

        if _stripe_key_cache:
            return StripePublishableKeyResponse(publishable_key=_stripe_key_cache)

        async with _stripe_key_lock:
            if _stripe_key_cache:
                return StripePublishableKeyResponse(publishable_key=_stripe_key_cache)

            override = app_settings.stripe_publishable_key_override
            if override:
                key = override.strip()
                if not key.startswith("pk_"):
                    raise HTTPException(
                        status_code=500, detail="Invalid Stripe publishable key format."
                    )
                _stripe_key_cache = key
                return StripePublishableKeyResponse(publishable_key=key)

            key_url = app_settings.stripe_publishable_key_url
            if not key_url:
                raise HTTPException(
                    status_code=500, detail="Stripe publishable key is not configured."
                )
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(key_url)
                    resp.raise_for_status()
                    key = resp.text.strip()
                if not key.startswith("pk_"):
                    raise HTTPException(
                        status_code=500, detail="Invalid Stripe publishable key format."
                    )
                _stripe_key_cache = key
                return StripePublishableKeyResponse(publishable_key=key)
            except httpx.HTTPError as e:
                raise HTTPException(
                    status_code=500, detail="Failed to fetch Stripe publishable key."
                ) from e

    @router.post("/reset-connection")
    async def reset_stripe_connection(
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> ResetConnectionResponse:
        """Close the in-memory circuit breaker to re-enable billing requests."""
        _close_billing_circuit()
        return ResetConnectionResponse(
            success=True,
            message="Stripe connection reset. Billing requests re-enabled.",
        )

    return router


class ResetConnectionResponse(BaseModel):
    success: bool
    message: str


__all__ = ["create_billing_router"]
