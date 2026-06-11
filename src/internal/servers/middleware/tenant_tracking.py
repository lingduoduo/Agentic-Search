"""Tenant tracking middleware — no-op stub for single-tenant deployments.

The previous version set CURRENT_TENANT_ID_CONTEXTVAR per request by
extracting tenant identity from Redis session tokens, API-key headers,
anonymous user JWT cookies, and PostgreSQL schema names. This requires
the full multi-tenant infrastructure (Redis, PostgreSQL multi-schema,
shared_configs contextvars) which does not exist in this single-tenant
self-hosted deployment.

In this repo every request implicitly belongs to the single local tenant.
The function is retained so call-sites that import it compile unchanged.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from src.internal.configs import AppSettings

logger = logging.getLogger(__name__)


def add_api_server_tenant_id_middleware(
    app: FastAPI,
    app_settings: AppSettings,  # noqa: ARG001
) -> None:
    """No-op — tenant tracking is not applicable to single-tenant deployments."""
    logger.debug("Tenant tracking middleware skipped (single-tenant deployment).")


__all__ = ["add_api_server_tenant_id_middleware"]
