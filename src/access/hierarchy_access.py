"""Helpers for propagating access across document hierarchies."""

from __future__ import annotations

from collections.abc import Iterable

from src.retrieval.models import DocumentAccess


def merge_document_access(
    own_access: DocumentAccess,
    inherited_access: Iterable[DocumentAccess] = (),
) -> DocumentAccess:
    """Combine direct and inherited permissions for a hierarchy node."""

    is_public = own_access.is_public
    user_ids = set(own_access.user_ids)
    group_ids = set(own_access.group_ids)
    for access in inherited_access:
        is_public = is_public or access.is_public
        user_ids.update(access.user_ids)
        group_ids.update(access.group_ids)
    return DocumentAccess(
        is_public=is_public,
        user_ids=user_ids,
        group_ids=group_ids,
    )


def inherited_group_ids(access_chain: Iterable[DocumentAccess]) -> set[str]:
    """Collect group IDs from ancestors for index-time metadata enrichment."""

    groups: set[str] = set()
    for access in access_chain:
        groups.update(access.group_ids)
    return groups


__all__ = ["inherited_group_ids", "merge_document_access"]
