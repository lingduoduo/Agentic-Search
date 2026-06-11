"""Analytics API router for Agentic Search.

Provides admin-only endpoints that aggregate chat session and user activity
stored in AgenticSearchStore. Endpoints mirror the structure of the sampled
Analytics API adapted to the data model available in this repo:

  - sessions  ↔  queries   (one session = one user query interaction)
  - messages  ↔  responses (message count within a session)
  - user_id   ↔  active user identity

Admin access is controlled by AppSettings.auth.super_users.
"""

from __future__ import annotations

import datetime

from fastapi import APIRouter
from fastapi import Depends
from pydantic import BaseModel

from src.backend.auth import AuthenticatedUser
from src.backend.configs import AppSettings
from src.backend.db import AgenticSearchStore
from src.backend.servers._auth import make_require_admin

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


class BreakdownItem(BaseModel):
    """Single dimension-value with its session count."""

    label: str
    session_count: int


class BreakdownAnalyticsResponse(BaseModel):
    """Session counts grouped by a single dimension (LLM, persona, or flow)."""

    dimension: str
    items: list[BreakdownItem]
    total_sessions: int


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_analytics_router(
    store: AgenticSearchStore,
    app_settings: AppSettings,
) -> APIRouter:
    """Return an APIRouter with analytics endpoints bound to *store*."""
    router = APIRouter(prefix="/analytics", tags=["analytics"])

    _require_admin = make_require_admin(app_settings)

    def _resolve_window(
        start: datetime.datetime | None,
        end: datetime.datetime | None,
    ) -> tuple[str, str]:
        default_start, default_end = _default_window()
        return (
            (start or default_start).isoformat(),
            (end or default_end).isoformat(),
        )

    @router.get("/query")
    def get_query_analytics(
        start: datetime.datetime | None = None,
        end: datetime.datetime | None = None,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> list[QueryAnalyticsResponse]:
        """Return daily session and message counts for the given date range."""
        s, e = _resolve_window(start, end)
        rows = store.query_session_analytics(s, e)
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
        s, e = _resolve_window(start, end)
        rows = store.user_session_analytics(s, e)
        return [
            UserAnalyticsResponse(
                date=datetime.date.fromisoformat(day),
                total_active_users=active_users,
            )
            for day, active_users in rows
        ]

    @router.get("/by-llm")
    def get_analytics_by_llm(
        start: datetime.datetime | None = None,
        end: datetime.datetime | None = None,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> BreakdownAnalyticsResponse:
        """Return session counts grouped by LLM model name."""
        s, e = _resolve_window(start, end)
        rows = store.query_analytics_by_llm(s, e)
        items = [
            BreakdownItem(label=label, session_count=count) for label, count in rows
        ]
        return BreakdownAnalyticsResponse(
            dimension="llm",
            items=items,
            total_sessions=sum(i.session_count for i in items),
        )

    @router.get("/by-persona")
    def get_analytics_by_persona(
        start: datetime.datetime | None = None,
        end: datetime.datetime | None = None,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> BreakdownAnalyticsResponse:
        """Return session counts grouped by persona / agent."""
        s, e = _resolve_window(start, end)
        rows = store.query_analytics_by_persona(s, e)
        items = [
            BreakdownItem(label=label, session_count=count) for label, count in rows
        ]
        return BreakdownAnalyticsResponse(
            dimension="persona",
            items=items,
            total_sessions=sum(i.session_count for i in items),
        )

    @router.get("/by-flow")
    def get_analytics_by_flow(
        start: datetime.datetime | None = None,
        end: datetime.datetime | None = None,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> BreakdownAnalyticsResponse:
        """Return session counts grouped by flow type (chat, slack, etc.)."""
        s, e = _resolve_window(start, end)
        rows = store.query_analytics_by_flow(s, e)
        items = [
            BreakdownItem(label=label, session_count=count) for label, count in rows
        ]
        return BreakdownAnalyticsResponse(
            dimension="flow",
            items=items,
            total_sessions=sum(i.session_count for i in items),
        )

    return router


__all__ = [
    "BreakdownAnalyticsResponse",
    "BreakdownItem",
    "QueryAnalyticsResponse",
    "UserAnalyticsResponse",
    "create_analytics_router",
]
