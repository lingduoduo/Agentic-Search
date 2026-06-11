"""Pydantic models for the user-group management API.

The models referenced ORM relationships (cc_pairs, personas,
document_sets, curator_ids) and used integer/UUID primary keys. This repo
uses string IDs and the group model only tracks membership — no personas,
connectors, or document-sets are associated with groups here.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.internal.db.models import GroupRecord


class UserGroupView(BaseModel):
    """API representation of a group and its current members."""

    id: str
    name: str
    user_ids: list[str]

    @classmethod
    def from_record(cls, record: GroupRecord) -> "UserGroupView":
        return cls(id=record.id, name=record.name, user_ids=list(record.user_ids))


class UserGroupCreate(BaseModel):
    name: str
    user_ids: list[str] = []
    group_id: str | None = None


class UserGroupUpdate(BaseModel):
    name: str | None = None
    user_ids: list[str] | None = None


class AddUsersToGroupRequest(BaseModel):
    user_ids: list[str]


class RemoveUsersFromGroupRequest(BaseModel):
    user_ids: list[str]


class UserGroupRename(BaseModel):
    name: str


__all__ = [
    "AddUsersToGroupRequest",
    "RemoveUsersFromGroupRequest",
    "UserGroupCreate",
    "UserGroupRename",
    "UserGroupUpdate",
    "UserGroupView",
]
