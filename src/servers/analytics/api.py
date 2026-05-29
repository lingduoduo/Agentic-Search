"""Analytics API router for Agentic Search.

Provides admin-only endpoints that aggregate chat session and user activity
stored in AgenticSearchStore. Endpoints mirror the structure of the sampled
onyx analytics API, adapted to the data model available in this repo:

  - sessions  ↔  queries   (one session = one user query interaction)
  - messages  ↔  responses (message count within a session)
  - user_id   ↔  active user identity

Admin access is controlled by AppSettings.auth.super_users.
"""

from __future__ import annotations

import datetime

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from pydantic import BaseModel

from src.auth import AuthenticatedUser
from src.auth import user_from_headers
from src.configs import AppSettings
from src.db import AgenticSearchStore

_DEFAULT_LOOKBACK_DAYS = 30


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.timezone.utc)


def _default_window() -> tuple[datetime.datetime, datetime.datetime]:
    end = _utc_now()
    start = end - datetime.timedelta(days=_DEFAULT_LOOKBACK_DAYS)
    return start, end


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class QueryAnalyticsResponse(BaseModel):
    """Daily session and message counts within the requested window."""

    date: datetime.date
    total_sessions: int
    total_messages: int


class UserAnalyticsResponse(BaseModel):
    """Daily count of distinct active users within the requested window."""

    date: datetime.date
    total_active_users: int


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_analytics_router(
    store: AgenticSearchStore,
    app_settings: AppSettings,
) -> APIRouter:
    """Return an APIRouter with analytics endpoints bound to *store*.

    Dependencies are injected at construction time so endpoints can access
    the store and settings without FastAPI's DI container.
    """
    router = APIRouter(prefix="/analytics", tags=["analytics"])

    def _require_admin(http_request: Request) -> AuthenticatedUser:
        user = user_from_headers(http_request.headers)
        if user is None or user.is_anonymous:
            raise HTTPException(status_code=401, detail="Authentication required.")
        super_users = app_settings.auth.super_users
        if user.id not in super_users and (
            user.email is None or user.email not in super_users
        ):
            raise HTTPException(status_code=403, detail="Admin access required.")
        return user

    @router.get("/query")
    def get_query_analytics(
        start: datetime.datetime | None = None,
        end: datetime.datetime | None = None,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> list[QueryAnalyticsResponse]:
        """Return daily session and message counts for the given date range."""
        default_start, default_end = _default_window()
        resolved_start = start or default_start
        resolved_end = end or default_end
        rows = store.query_session_analytics(
            resolved_start.isoformat(),
            resolved_end.isoformat(),
        )
        return [
            QueryAnalyticsResponse(
                date=datetime.date.fromisoformat(day),
                total_sessions=sessions,
                total_messages=messages,
            )
            for day, sessions, messages in rows
        ]

    @router.get("/user")
    def get_user_analytics(
        start: datetime.datetime | None = None,
        end: datetime.datetime | None = None,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> list[UserAnalyticsResponse]:
        """Return daily distinct active user counts for the given date range."""
        default_start, default_end = _default_window()
        resolved_start = start or default_start
        resolved_end = end or default_end
        rows = store.user_session_analytics(
            resolved_start.isoformat(),
            resolved_end.isoformat(),
        )
        return [
            UserAnalyticsResponse(
                date=datetime.date.fromisoformat(day),
                total_active_users=active_users,
            )
            for day, active_users in rows
        ]

    return router


__all__ = [
    "QueryAnalyticsResponse",
    "UserAnalyticsResponse",
    "create_analytics_router",
]
