"""Lightweight access-filter helpers for local search contexts."""

from __future__ import annotations

from ..models import SearchFilters


def build_access_filter(principal_id: str | None) -> list[str] | None:
    if not principal_id:
        return None
    return [f"user:{principal_id}", "public"]


def build_user_only_filters(principal_id: str | None) -> SearchFilters:
    acl = build_access_filter(principal_id)
    return SearchFilters(tags={"acl": ",".join(acl)} if acl else None)


__all__ = ["build_access_filter", "build_user_only_filters"]
