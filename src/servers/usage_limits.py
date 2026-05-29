"""Usage limit constants and structures for Agentic Search.

This repo is a single-tenant self-hosted deployment. All limits default to
NO_LIMIT (-1, sentinel for unlimited). The ``TenantUsageLimitOverrides``
dataclass mirrors the structure from the sampled multi-tenant code so callers
can work with the same field names without modification.
"""

from __future__ import annotations

from dataclasses import dataclass

NO_LIMIT: int = -1


@dataclass
class TenantUsageLimitOverrides:
    """Per-tenant usage limit overrides. ``NO_LIMIT`` means unlimited."""

    tenant_id: str
    llm_cost_cents_trial: int = NO_LIMIT
    llm_cost_cents_paid: int = NO_LIMIT
    chunks_indexed_trial: int = NO_LIMIT
    chunks_indexed_paid: int = NO_LIMIT
    api_calls_trial: int = NO_LIMIT
    api_calls_paid: int = NO_LIMIT
    non_streaming_calls_trial: int = NO_LIMIT
    non_streaming_calls_paid: int = NO_LIMIT

    def is_unlimited(self) -> bool:
        return all(
            v == NO_LIMIT
            for v in (
                self.llm_cost_cents_trial,
                self.llm_cost_cents_paid,
                self.chunks_indexed_trial,
                self.chunks_indexed_paid,
                self.api_calls_trial,
                self.api_calls_paid,
                self.non_streaming_calls_trial,
                self.non_streaming_calls_paid,
            )
        )


def is_tenant_on_trial(tenant_id: str) -> bool:  # noqa: ARG001
    """Return whether *tenant_id* is on a trial subscription.

    This repo is single-tenant and has no billing infrastructure, so
    this always returns ``False``.
    """
    return False


__all__ = [
    "NO_LIMIT",
    "TenantUsageLimitOverrides",
    "is_tenant_on_trial",
]
