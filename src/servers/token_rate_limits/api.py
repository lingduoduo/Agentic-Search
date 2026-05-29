"""Token rate limit admin API.

Adapted from the sampled Onyx ee/onyx/server/token_rate_limits/api.py.
SQLAlchemy ORM and ee.onyx/onyx imports are replaced with AgenticSearchStore.
Group IDs are strings (store primary keys) rather than integers.
"""

from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from src.auth import AuthenticatedUser
from src.configs import AppSettings
from src.db import AgenticSearchStore
from src.servers.token_rate_limits.models import TokenRateLimitArgs
from src.servers.token_rate_limits.models import TokenRateLimitDisplay
from src.servers._auth import make_require_admin


def create_token_rate_limits_router(
    store: AgenticSearchStore,
    app_settings: AppSettings,
) -> APIRouter:
    """Return an APIRouter with token-rate-limit admin endpoints."""

    router = APIRouter(prefix="/admin/token-rate-limits", tags=["token-rate-limits"])

    _require_admin = make_require_admin(app_settings)

    # -----------------------------------------------------------------------
    # Group token limit endpoints
    # -----------------------------------------------------------------------

    @router.get("/user-groups")
    def get_all_group_token_limit_settings(
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> dict[str, list[TokenRateLimitDisplay]]:
        """Return all group-scoped token rate limits keyed by group name."""
        records = store.get_all_group_token_rate_limits()
        by_group: dict[str, list[TokenRateLimitDisplay]] = defaultdict(list)
        for record in records:
            group_name = str(
                record.get("group_name") or record.get("scope_id") or "unknown"
            )
            by_group[group_name].append(TokenRateLimitDisplay.from_record(record))
        return dict(by_group)

    @router.get("/user-group/{group_id}")
    def get_group_token_limit_settings(
        group_id: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> list[TokenRateLimitDisplay]:
        """Return token rate limits for a specific group."""
        group = store.get_group(group_id)
        if group is None:
            raise HTTPException(
                status_code=404, detail=f"Group '{group_id}' not found."
            )
        records = store.get_token_rate_limits(scope="group", scope_id=group_id)
        return [TokenRateLimitDisplay.from_record(r) for r in records]

    @router.post("/user-group/{group_id}", status_code=201)
    def create_group_token_limit_settings(
        group_id: str,
        token_limit_settings: TokenRateLimitArgs,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> TokenRateLimitDisplay:
        """Create a token rate limit rule for a specific group."""
        if store.get_group(group_id) is None:
            raise HTTPException(
                status_code=404, detail=f"Group '{group_id}' not found."
            )
        record = store.insert_token_rate_limit(
            scope="group",
            scope_id=group_id,
            token_budget=token_limit_settings.token_budget,
            period_hours=token_limit_settings.period_hours,
            enabled=token_limit_settings.enabled,
        )
        return TokenRateLimitDisplay.from_record(record)

    # -----------------------------------------------------------------------
    # Global user token limit endpoints
    # -----------------------------------------------------------------------

    @router.get("/users")
    def get_user_token_limit_settings(
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> list[TokenRateLimitDisplay]:
        """Return global user-scoped token rate limits."""
        records = store.get_token_rate_limits(scope="user")
        return [TokenRateLimitDisplay.from_record(r) for r in records]

    @router.post("/users", status_code=201)
    def create_user_token_limit_settings(
        token_limit_settings: TokenRateLimitArgs,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> TokenRateLimitDisplay:
        """Create a global token rate limit that applies to all users."""
        record = store.insert_token_rate_limit(
            scope="user",
            scope_id=None,
            token_budget=token_limit_settings.token_budget,
            period_hours=token_limit_settings.period_hours,
            enabled=token_limit_settings.enabled,
        )
        return TokenRateLimitDisplay.from_record(record)

    return router


__all__ = ["create_token_rate_limits_router"]
