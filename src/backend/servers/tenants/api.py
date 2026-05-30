"""Tenants API aggregator.

This repo is a single-tenant self-hosted deployment. Multi-tenant features
(billing, provisioning, schema management, anonymous users, SCIM, impersonation)
require the full Onyx cloud infrastructure and are not implemented here.

The router exists so that code that imports from this module compiles; all
endpoints return HTTP 501 Not Implemented.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi import HTTPException

router = APIRouter(prefix="/tenants", tags=["tenants"])


def _not_implemented() -> None:
    raise HTTPException(
        status_code=501,
        detail=(
            "Multi-tenant features are not available in this self-hosted deployment."
        ),
    )


@router.get("/billing-information")
def get_billing_information() -> None:
    """Cloud billing feature — not available in self-hosted."""
    _not_implemented()


@router.post("/create-subscription-session")
def create_subscription_session() -> None:
    """Cloud billing feature — not available in self-hosted."""
    _not_implemented()


@router.post("/create-customer-portal-session")
def create_customer_portal_session() -> None:
    """Cloud billing feature — not available in self-hosted."""
    _not_implemented()


@router.get("/stripe-publishable-key")
def get_stripe_publishable_key() -> None:
    """Cloud billing feature — not available in self-hosted."""
    _not_implemented()


__all__ = ["router"]
