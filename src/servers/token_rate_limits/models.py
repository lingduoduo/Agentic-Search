"""Pydantic models for the token rate limit API."""

from __future__ import annotations

from pydantic import BaseModel


class TokenRateLimitArgs(BaseModel):
    """Configuration for a new token rate limit rule."""

    enabled: bool = True
    token_budget: int
    period_hours: int


class TokenRateLimitDisplay(BaseModel):
    """A stored token rate limit rule returned by the API."""

    id: str
    enabled: bool
    token_budget: int
    period_hours: int
    scope: str
    scope_id: str | None

    @classmethod
    def from_record(cls, record: dict) -> "TokenRateLimitDisplay":
        return cls(
            id=str(record["id"]),
            enabled=bool(record["enabled"]),
            token_budget=int(record["token_budget"]),
            period_hours=int(record["period_hours"]),
            scope=str(record["scope"]),
            scope_id=record.get("scope_id"),
        )


__all__ = ["TokenRateLimitArgs", "TokenRateLimitDisplay"]
