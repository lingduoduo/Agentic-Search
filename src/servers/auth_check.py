"""Route authentication audit helpers.

``check_router_auth`` iterates every registered FastAPI route and logs the
auth posture of each one relative to a declared public-endpoint allowlist.
This makes accidental unauthenticated routes visible at startup without
requiring middleware to enforce auth (the repo uses per-handler checks via
``user_from_headers``).

Usage:
    from src.servers.auth_check import check_router_auth, PUBLIC_ENDPOINT_SPECS
    check_router_auth(app, PUBLIC_ENDPOINT_SPECS)
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from starlette.routing import Route

logger = logging.getLogger(__name__)

# Endpoints that intentionally require no authentication.
PUBLIC_ENDPOINT_SPECS: list[tuple[str, set[str]]] = [
    ("/health", {"GET"}),
    ("/", {"GET"}),
    ("/assets/app.css", {"GET"}),
    ("/assets/app.js", {"GET"}),
    # Static assets directory (mounted)
    ("/assets", {"GET", "HEAD"}),
]

# Analytics and agent endpoints added for EE / extended deployments.
EE_PUBLIC_ENDPOINT_SPECS: list[tuple[str, set[str]]] = PUBLIC_ENDPOINT_SPECS + [
    # SCIM 2.0 service discovery — unauthenticated so IdPs can probe.
    ("/scim/v2/ServiceProviderConfig", {"GET"}),
    ("/scim/v2/ResourceTypes", {"GET"}),
    ("/scim/v2/Schemas", {"GET"}),
    # Stripe publishable key is safe to expose publicly.
    ("/tenants/stripe-publishable-key", {"GET"}),
    # Proxy billing endpoints use license-based auth.
    ("/proxy/create-checkout-session", {"POST"}),
    ("/proxy/claim-license", {"POST"}),
    ("/proxy/create-customer-portal-session", {"POST"}),
    ("/proxy/billing-information", {"GET"}),
]


def _is_public(
    path: str,
    methods: set[str],
    public_specs: list[tuple[str, set[str]]],
) -> bool:
    for public_path, public_methods in public_specs:
        if path == public_path and methods & public_methods:
            return True
    return False


def check_router_auth(
    application: FastAPI,
    public_endpoint_specs: list[tuple[str, set[str]]] = PUBLIC_ENDPOINT_SPECS,
) -> None:
    """Audit all registered routes against *public_endpoint_specs*.

    Routes not in the public list are logged at DEBUG (expected to have
    per-handler auth). Routes in the public list are logged at INFO so the
    set of intentionally unauthenticated paths is visible in startup logs.
    Missing routes from the public list (declared public but not registered)
    are logged as warnings — this catches stale entries in the allowlist.
    """
    registered: dict[str, set[str]] = {}
    for route in application.routes:
        if isinstance(route, Route) and route.methods:
            registered.setdefault(route.path, set()).update(route.methods)

    # Check every registered route.
    for path, methods in sorted(registered.items()):
        if _is_public(path, methods, public_endpoint_specs):
            logger.info("[auth_check] public  %s %s", sorted(methods), path)
        else:
            logger.debug("[auth_check] guarded %s %s", sorted(methods), path)

    # Warn about public specs that don't match any registered route.
    for public_path, public_methods in public_endpoint_specs:
        if public_path not in registered:
            logger.warning(
                "[auth_check] declared public but not registered: %s %s",
                sorted(public_methods),
                public_path,
            )


def check_ee_router_auth(
    application: FastAPI,
    public_endpoint_specs: list[tuple[str, set[str]]] = EE_PUBLIC_ENDPOINT_SPECS,
) -> None:
    """Variant of ``check_router_auth`` that includes EE public endpoint specs."""
    check_router_auth(application, public_endpoint_specs)


__all__ = [
    "EE_PUBLIC_ENDPOINT_SPECS",
    "PUBLIC_ENDPOINT_SPECS",
    "check_ee_router_auth",
    "check_router_auth",
]
