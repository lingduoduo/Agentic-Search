"""Access-control helpers shared by local search and web APIs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.internal.auth import AuthenticatedUser
from src.internal.db import AgenticSearchStore
from src.internal.document_index.models import DocumentAccess

PUBLIC_ACL = "public"
USER_PREFIX = "user:"
EMAIL_PREFIX = "email:"
GROUP_PREFIX = "group:"
EXTERNAL_GROUP_PREFIX = "external_group:"


def prefix_user(user_id: str) -> str:
    return f"{USER_PREFIX}{user_id}"


def prefix_email(email: str) -> str:
    return f"{EMAIL_PREFIX}{email.lower()}"


def prefix_group(group_id: str) -> str:
    return f"{GROUP_PREFIX}{group_id}"


def prefix_external_group(group_id: str) -> str:
    return f"{EXTERNAL_GROUP_PREFIX}{group_id}"


def acl_for_user(user: AuthenticatedUser | None) -> set[str]:
    """Return ACL entries that should grant visibility to *user*."""

    entries = {PUBLIC_ACL}
    if user is None or user.is_anonymous:
        return entries
    entries.add(prefix_user(user.id))
    if user.email:
        entries.add(prefix_email(user.email))
    entries.update(prefix_group(group_id) for group_id in user.group_ids)
    entries.update(
        prefix_external_group(group_id)
        for group_id in _external_group_ids_from_metadata(user.metadata)
    )
    return entries


def acl_for_store_user(
    store: AgenticSearchStore,
    user_id: str | None,
) -> set[str]:
    """Build ACL entries from the local metadata store."""

    entries = {PUBLIC_ACL}
    if user_id is None:
        return entries
    entries.add(prefix_user(user_id))
    user = store.get_user(user_id)
    if user and user.email:
        entries.add(prefix_email(user.email))
    groups = store.list_groups_for_user(user_id)
    entries.update(prefix_group(group.id) for group in groups)
    for group in groups:
        mapping = store.get_scim_group_mapping(group.id)
        if mapping and mapping.get("external_id"):
            entries.add(prefix_external_group(str(mapping["external_id"])))
    return entries


def acl_for_document_access(access: DocumentAccess) -> set[str]:
    entries = {PUBLIC_ACL} if access.is_public else set()
    entries.update(prefix_user(user_id) for user_id in access.user_ids)
    entries.update(prefix_group(group_id) for group_id in access.group_ids)
    return entries


def can_access_acl(
    document_acl: Iterable[str],
    user_acl: Iterable[str],
) -> bool:
    user_entries = user_acl if isinstance(user_acl, set) else set(user_acl)
    return any(entry in user_entries for entry in document_acl)


def can_access_document(
    access: DocumentAccess,
    user: AuthenticatedUser | None,
) -> bool:
    return can_access_acl(acl_for_document_access(access), acl_for_user(user))


def metadata_with_acl(
    metadata: dict[str, object] | None = None,
    *,
    access: DocumentAccess | None = None,
    acl: Iterable[str] | None = None,
) -> dict[str, object]:
    """Return a metadata copy with an ``acl`` list suitable for search filters."""

    enriched = dict(metadata or {})
    if acl is not None:
        enriched["acl"] = sorted(set(acl))
    elif access is not None:
        enriched["acl"] = sorted(acl_for_document_access(access))
    return enriched


def _external_group_ids_from_metadata(metadata: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("external_group_ids", "external_groups"):
        raw = metadata.get(key)
        if raw is None:
            continue
        if isinstance(raw, str):
            values.update(part.strip() for part in raw.split(",") if part.strip())
        elif isinstance(raw, Iterable):
            values.update(str(part).strip() for part in raw if str(part).strip())
        else:
            values.add(str(raw))
    return values


__all__ = [
    "EMAIL_PREFIX",
    "EXTERNAL_GROUP_PREFIX",
    "GROUP_PREFIX",
    "PUBLIC_ACL",
    "USER_PREFIX",
    "acl_for_document_access",
    "acl_for_store_user",
    "acl_for_user",
    "can_access_acl",
    "can_access_document",
    "metadata_with_acl",
    "prefix_email",
    "prefix_external_group",
    "prefix_group",
    "prefix_user",
]
