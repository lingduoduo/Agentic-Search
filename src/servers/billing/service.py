"""Billing service layer.

Adapted from the sampled Onyx ee/onyx/server/billing/service.py.
Changes from the original:
- MULTI_TENANT removed — this is a single-tenant self-hosted deployment;
  requests always proxy through cloud_data_plane_url (never direct control plane).
- OnyxError / OnyxErrorCode replaced with standard FastAPI HTTPException.
- generate_data_plane_token (EE JWT) replaced with license-bearer auth.
- setup_logger replaced with standard logging.getLogger.
- CLOUD_DATA_PLANE_URL / CONTROL_PLANE_API_BASE_URL replaced with
  a runtime-injected base_url parameter.
"""

from __future__ import annotations

import logging
from typing import Literal

import httpx
from fastapi import HTTPException

from src.servers.billing.models import BillingInformationResponse
from src.servers.billing.models import CreateCheckoutSessionResponse
from src.servers.billing.models import CreateCustomerPortalSessionResponse
from src.servers.billing.models import EndTrialResponse
from src.servers.billing.models import SeatUpdateResponse
from src.servers.billing.models import StripePortalFlowType
from src.servers.billing.models import SubscriptionStatusResponse

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 30.0


async def _make_billing_request(
    method: Literal["GET", "POST"],
    path: str,
    base_url: str,
    license_data: str | None = None,
    body: dict | None = None,
    params: dict | None = None,
    error_message: str = "Billing service request failed",
) -> dict:
    """Make an HTTP request to the billing proxy.

    Args:
        method: HTTP method.
        path: Path appended to *base_url*.
        base_url: Cloud data plane proxy URL (e.g. ``https://cloud.example.com/proxy``).
        license_data: Bearer token (license blob) for self-hosted auth.
        body: JSON body for POST requests.
        params: Query parameters for GET requests.
        error_message: Default message if the request fails.
    """
    url = f"{base_url.rstrip('/')}{path}"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if license_data:
        headers["Authorization"] = f"Bearer {license_data}"

    try:
        async with httpx.AsyncClient(
            timeout=_REQUEST_TIMEOUT, follow_redirects=True
        ) as client:
            if method == "GET":
                response = await client.get(url, headers=headers, params=params)
            else:
                response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return response.json()

    except httpx.HTTPStatusError as e:
        detail = error_message
        try:
            error_data = e.response.json()
            detail = error_data.get("detail", detail)
        except Exception:
            pass
        logger.error("%s: %s — %s", error_message, e.response.status_code, detail)
        raise HTTPException(status_code=e.response.status_code, detail=detail) from e

    except httpx.RequestError as e:
        logger.exception("Failed to connect to billing service")
        raise HTTPException(
            status_code=502, detail="Failed to connect to billing service"
        ) from e


async def create_checkout_session(
    base_url: str,
    billing_period: str = "monthly",
    seats: int | None = None,
    email: str | None = None,
    license_data: str | None = None,
    redirect_url: str | None = None,
) -> CreateCheckoutSessionResponse:
    body: dict = {"billing_period": billing_period}
    if seats is not None:
        body["seats"] = seats
    if email:
        body["email"] = email
    if redirect_url:
        body["redirect_url"] = redirect_url

    data = await _make_billing_request(
        method="POST",
        path="/create-checkout-session",
        base_url=base_url,
        license_data=license_data,
        body=body,
        error_message="Failed to create checkout session",
    )
    return CreateCheckoutSessionResponse(stripe_checkout_url=data["url"])


async def create_customer_portal_session(
    base_url: str,
    license_data: str | None = None,
    return_url: str | None = None,
    flow_type: StripePortalFlowType | None = None,
) -> CreateCustomerPortalSessionResponse:
    body: dict = {}
    if return_url:
        body["return_url"] = return_url
    if flow_type:
        body["flow_type"] = flow_type.value

    data = await _make_billing_request(
        method="POST",
        path="/create-customer-portal-session",
        base_url=base_url,
        license_data=license_data,
        body=body,
        error_message="Failed to create customer portal session",
    )
    return CreateCustomerPortalSessionResponse(stripe_customer_portal_url=data["url"])


async def get_billing_information(
    base_url: str,
    license_data: str | None = None,
) -> BillingInformationResponse | SubscriptionStatusResponse:
    data = await _make_billing_request(
        method="GET",
        path="/billing-information",
        base_url=base_url,
        license_data=license_data,
        error_message="Failed to fetch billing information",
    )
    if isinstance(data, dict) and data.get("subscribed") is False:
        return SubscriptionStatusResponse(subscribed=False)
    return BillingInformationResponse(**data)


async def update_seat_count(
    base_url: str,
    new_seat_count: int,
    license_data: str | None = None,
) -> SeatUpdateResponse:
    data = await _make_billing_request(
        method="POST",
        path="/seats/update",
        base_url=base_url,
        license_data=license_data,
        body={"new_seat_count": new_seat_count},
        error_message="Failed to update seat count",
    )
    return SeatUpdateResponse(
        success=data.get("success", False),
        current_seats=data.get("current_seats", 0),
        used_seats=data.get("used_seats", 0),
        message=data.get("message"),
        license=data.get("license"),
    )


async def end_trial() -> EndTrialResponse:
    """End trial — cloud-only; not supported in self-hosted."""
    raise HTTPException(
        status_code=501,
        detail="End-trial is only available for cloud deployments.",
    )


__all__ = [
    "create_checkout_session",
    "create_customer_portal_session",
    "end_trial",
    "get_billing_information",
    "update_seat_count",
]
