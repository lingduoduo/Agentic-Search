"""Tenant usage limit resolution for Agentic Search.

The sampled EE version fetched overrides from a control-plane API with a
24-hour TTL cache backed by Redis. This repo is single-tenant and has no
control plane, so all tenants always receive unlimited limits.

The public interface (``get_tenant_usage_limit_overrides``, ``unlimited``) is
preserved so callers written against the sampled code work unchanged.
"""

from __future__ import annotations

import logging

from src.internal.servers.limits.usage_limits import NO_LIMIT
from src.internal.servers.limits.usage_limits import TenantUsageLimitOverrides

logger = logging.getLogger(__name__)


def unlimited(tenant_id: str) -> TenantUsageLimitOverrides:
    """Return an unlimited ``TenantUsageLimitOverrides`` for *tenant_id*."""
    return TenantUsageLimitOverrides(
        tenant_id=tenant_id,
        llm_cost_cents_trial=NO_LIMIT,
        llm_cost_cents_paid=NO_LIMIT,
        chunks_indexed_trial=NO_LIMIT,
        chunks_indexed_paid=NO_LIMIT,
        api_calls_trial=NO_LIMIT,
        api_calls_paid=NO_LIMIT,
        non_streaming_calls_trial=NO_LIMIT,
        non_streaming_calls_paid=NO_LIMIT,
    )


def get_tenant_usage_limit_overrides(
    tenant_id: str,
) -> TenantUsageLimitOverrides:
    """Return usage limit overrides for *tenant_id*.

    This repo is single-tenant; all callers receive unlimited limits. In a
    multi-tenant deployment this would fetch from a control-plane API with a
    cache TTL (see the sampled ee version).
    """
    return unlimited(tenant_id)


__all__ = [
    "TenantUsageLimitOverrides",
    "get_tenant_usage_limit_overrides",
    "unlimited",
]
