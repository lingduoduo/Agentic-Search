"""Allowlist helpers for tenants with inactive subscriptions."""

from __future__ import annotations

from .license_enforcement_config import path_matches_prefix

MULTI_TENANT_GATING_ALLOWED_PREFIXES: frozenset[str] = frozenset(
    {
        "/auth",
        "/health",
        "/me",
        "/settings",
        "/billing",
        "/admin/billing",
        "/api/sessions",
        "/assets",
        "/tenants/billing-information",
        "/tenants/create-customer-portal-session",
        "/tenants/create-subscription-session",
        "/tenants/stripe-publishable-key",
    }
)


def is_path_allowed_for_gated_tenant(path: str) -> bool:
    return any(
        path_matches_prefix(path, prefix)
        for prefix in MULTI_TENANT_GATING_ALLOWED_PREFIXES
    )


__all__ = [
    "MULTI_TENANT_GATING_ALLOWED_PREFIXES",
    "is_path_allowed_for_gated_tenant",
]
