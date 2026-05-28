"""Lightweight access-filter helpers for local search contexts."""

from __future__ import annotations

from collections.abc import Iterable

from src.access import PUBLIC_ACL
from src.access import prefix_email
from src.access import prefix_group
from src.access import prefix_user

from ..models import SearchFilters


def build_access_filter(
    principal_id: str | None = None,
    *,
    email: str | None = None,
    group_ids: Iterable[str] | None = None,
    include_public: bool = True,
) -> list[str]:
    acl: set[str] = {PUBLIC_ACL} if include_public else set()
    if principal_id:
        acl.add(prefix_user(principal_id))
    if email:
        acl.add(prefix_email(email))
    acl.update(prefix_group(group_id) for group_id in group_ids or [])
    return sorted(acl)


def build_user_only_filters(
    principal_id: str | None,
    *,
    email: str | None = None,
    group_ids: Iterable[str] | None = None,
) -> SearchFilters:
    return SearchFilters(
        access_acl=build_access_filter(
            principal_id,
            email=email,
            group_ids=group_ids,
        )
    )


__all__ = ["build_access_filter", "build_user_only_filters"]
