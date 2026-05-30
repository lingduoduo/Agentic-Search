"""SCIM Data Access Layer backed by AgenticSearchStore.

Replaces the SQLAlchemy-based ee.onyx.db.scim.ScimDAL. Provides SCIM-specific
queries over the project's SQLite store. Seat-checking and permission
recomputation (EE/PostgreSQL-only) are omitted.
"""

from __future__ import annotations

from src.backend.db import AgenticSearchStore
from src.backend.servers.scim.filtering import ScimFilter
from src.backend.servers.scim.filtering import ScimFilterOperator
from src.backend.servers.scim.models import ScimMappingFields
from src.backend.servers.scim.providers.base import ScimGroup
from src.backend.servers.scim.providers.base import ScimUser


def _apply_filter(
    rows: list[dict], scim_filter: ScimFilter | None, attr_map: dict
) -> list[dict]:
    """Apply a SCIM filter to a list of dicts in Python."""
    if not scim_filter:
        return rows
    attr = scim_filter.attribute.lower()
    field = attr_map.get(attr)
    if not field:
        return rows
    val = scim_filter.value.lower()
    filtered = []
    for row in rows:
        row_val = str(row.get(field) or "").lower()
        if scim_filter.operator == ScimFilterOperator.EQUAL and row_val == val:
            filtered.append(row)
        elif scim_filter.operator == ScimFilterOperator.CONTAINS and val in row_val:
            filtered.append(row)
        elif (
            scim_filter.operator == ScimFilterOperator.STARTS_WITH
            and row_val.startswith(val)
        ):
            filtered.append(row)
    return filtered


class ScimDAL:
    """SCIM data access backed by AgenticSearchStore."""

    def __init__(self, store: AgenticSearchStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def get_token_by_hash(self, token_hash: str) -> dict | None:
        return self._store.get_scim_token_by_hash(token_hash)

    def update_token_last_used(self, token_id: str) -> None:
        self._store.update_scim_token_last_used(token_id)

    def create_token(self, name: str, token_hash: str, token_display: str) -> dict:
        return self._store.create_scim_token(name, token_hash, token_display)

    def list_tokens(self) -> list[dict]:
        return self._store.list_scim_tokens()

    def revoke_token(self, token_id: str) -> bool:
        return self._store.revoke_scim_token(token_id)

    # ------------------------------------------------------------------
    # User operations
    # ------------------------------------------------------------------

    def list_users(
        self,
        scim_filter: ScimFilter | None,
        start_index: int,
        count: int,
    ) -> tuple[list[tuple[ScimUser, dict | None]], int]:
        """Return paginated users with their SCIM mappings."""
        user_records = self._store.list_all_users()
        rows = [
            {"id": u.id, "email": u.email or "", "name": u.name} for u in user_records
        ]
        attr_map = {"username": "email", "email": "email"}
        rows = _apply_filter(rows, scim_filter, attr_map)
        total = len(rows)
        page = rows[start_index - 1 : start_index - 1 + count]

        # Batch-load is_active and mappings for the page only
        page_ids = [r["id"] for r in page]
        active_map = {uid: self._store.get_user_active(uid) for uid in page_ids}

        results: list[tuple[ScimUser, dict | None]] = []
        for r in page:
            user = ScimUser(
                id=r["id"],
                email=r["email"],
                is_active=active_map.get(r["id"], True),
                personal_name=r.get("name"),
            )
            mapping = self._store.get_scim_user_mapping(r["id"])
            results.append((user, mapping))
        return results, total

    def get_user(self, user_id: str) -> ScimUser | None:
        rec = self._store.get_user(user_id)
        if not rec:
            return None
        return ScimUser(
            id=rec.id,
            email=rec.email or "",
            is_active=self._store.get_user_active(rec.id),
            personal_name=rec.name,
        )

    def get_user_by_email(self, email: str) -> ScimUser | None:
        rec = self._store.get_user_by_email(email)
        if not rec:
            return None
        return ScimUser(
            id=rec.id,
            email=rec.email or "",
            is_active=self._store.get_user_active(rec.id),
            personal_name=rec.name,
        )

    def add_user(self, user: ScimUser) -> None:
        from src.backend.db.models import UserRecord

        self._store.upsert_user(
            UserRecord(id=user.id, email=user.email, name=user.personal_name)
        )
        self._store.set_user_active(user.id, user.is_active)

    def update_user(
        self,
        user: ScimUser,
        email: str | None = None,
        is_active: bool | None = None,
        personal_name: str | None = None,
    ) -> None:
        from src.backend.db.models import UserRecord

        self._store.upsert_user(
            UserRecord(
                id=user.id,
                email=email if email is not None else user.email,
                name=personal_name if personal_name is not None else user.personal_name,
            )
        )
        if is_active is not None:
            self._store.set_user_active(user.id, is_active)
            user.is_active = is_active

    def deactivate_user(self, user: ScimUser) -> None:
        self._store.set_user_active(user.id, False)

    def get_user_mapping(self, user_id: str) -> dict | None:
        return self._store.get_scim_user_mapping(user_id)

    def create_user_mapping(
        self,
        external_id: str | None,
        user_id: str,
        scim_username: str | None = None,
        fields: ScimMappingFields | None = None,
    ) -> None:
        f = fields or ScimMappingFields()
        self._store.create_scim_user_mapping(
            user_id=user_id,
            external_id=external_id,
            scim_username=scim_username,
            department=f.department,
            manager=f.manager,
            given_name=f.given_name,
            family_name=f.family_name,
            scim_emails_json=f.scim_emails_json,
        )

    def delete_user_mapping(self, user_id: str) -> None:
        self._store.delete_scim_user_mapping(user_id)

    def sync_user_external_id(
        self,
        user_id: str,
        external_id: str | None,
        scim_username: str | None = None,
        fields: ScimMappingFields | None = None,
    ) -> None:
        f = fields or ScimMappingFields()
        self._store.upsert_scim_user_mapping(
            user_id=user_id,
            external_id=external_id,
            scim_username=scim_username,
            department=f.department,
            manager=f.manager,
            given_name=f.given_name,
            family_name=f.family_name,
            scim_emails_json=f.scim_emails_json,
        )

    def get_user_groups(self, user_id: str) -> list[tuple[str, str | None]]:
        """Return (group_id, group_name) pairs for a user."""
        groups = self._store.list_groups_for_user(user_id)
        return [(g.id, g.name) for g in groups]

    def get_users_groups_batch(
        self, user_ids: list[str]
    ) -> dict[str, list[tuple[str, str | None]]]:
        return self._store.get_groups_for_users_batch(user_ids)

    def validate_member_ids(self, user_ids: list[str]) -> list[str]:
        """Return IDs that do not exist in the store (single batch query)."""
        found = set(self._store.get_users_by_ids(user_ids))
        return [uid for uid in user_ids if uid not in found]

    # ------------------------------------------------------------------
    # Group operations
    # ------------------------------------------------------------------

    def list_groups(
        self,
        scim_filter: ScimFilter | None,
        start_index: int,
        count: int,
    ) -> tuple[list[tuple[ScimGroup, str | None]], int]:
        groups = self._store.list_groups()
        rows = [{"id": g.id, "name": g.name} for g in groups]
        attr_map = {"displayname": "name"}
        rows = _apply_filter(rows, scim_filter, attr_map)
        total = len(rows)
        page = rows[start_index - 1 : start_index - 1 + count]

        results: list[tuple[ScimGroup, str | None]] = []
        for r in page:
            group = ScimGroup(id=r["id"], name=r["name"])
            mapping = self._store.get_scim_group_mapping(r["id"])
            ext_id = mapping["external_id"] if mapping else None
            results.append((group, ext_id))
        return results, total

    def get_group(self, group_id: str) -> ScimGroup | None:
        rec = self._store.get_group(group_id)
        return ScimGroup(id=rec.id, name=rec.name) if rec else None

    def get_group_by_name(self, name: str) -> ScimGroup | None:
        for g in self._store.list_groups():
            if g.name == name:
                return ScimGroup(id=g.id, name=g.name)
        return None

    def get_group_members(self, group_id: str) -> list[tuple[str, str | None]]:
        """Return (user_id, email) pairs for a group (single batch query)."""
        rec = self._store.get_group(group_id)
        if not rec:
            return []
        emails = self._store.get_users_emails_batch(rec.user_ids)
        return [(uid, emails.get(uid)) for uid in rec.user_ids]

    def get_group_mapping(self, group_id: str) -> dict | None:
        return self._store.get_scim_group_mapping(group_id)

    def add_group(self, group: ScimGroup) -> None:
        from src.backend.db.models import GroupRecord

        self._store.upsert_group(GroupRecord(id=group.id, name=group.name))

    def create_group(self, name: str) -> ScimGroup:
        from src.backend.db.models import GroupRecord
        import uuid

        group_id = str(uuid.uuid4())
        self._store.upsert_group(GroupRecord(id=group_id, name=name))
        return ScimGroup(id=group_id, name=name)

    def update_group(self, group: ScimGroup, name: str | None = None) -> None:
        from src.backend.db.models import GroupRecord

        rec = self._store.get_group(group.id)
        current_members = rec.user_ids if rec else []
        self._store.upsert_group(
            GroupRecord(id=group.id, name=name or group.name, user_ids=current_members)
        )

    def delete_group(self, group: ScimGroup) -> None:
        self._store.delete_group(group.id)

    def set_group_members(self, group_id: str, user_ids: list[str]) -> None:
        from src.backend.db.models import GroupRecord

        rec = self._store.get_group(group_id)
        name = rec.name if rec else group_id
        self._store.upsert_group(GroupRecord(id=group_id, name=name, user_ids=user_ids))

    def add_group_members(self, group_id: str, user_ids: list[str]) -> None:
        rec = self._store.get_group(group_id)
        if not rec:
            return
        existing = set(rec.user_ids)
        existing.update(user_ids)
        self.set_group_members(group_id, list(existing))

    def remove_group_members(self, group_id: str, user_ids: list[str]) -> None:
        rec = self._store.get_group(group_id)
        if not rec:
            return
        remaining = [uid for uid in rec.user_ids if uid not in user_ids]
        self.set_group_members(group_id, remaining)

    def create_group_mapping(
        self, group_id: str, external_id: str | None = None
    ) -> None:
        self._store.create_scim_group_mapping(
            group_id=group_id, external_id=external_id
        )

    def delete_group_mapping(self, group_id: str) -> None:
        self._store.delete_scim_group_mapping(group_id)

    def sync_group_external_id(self, group_id: str, external_id: str | None) -> None:
        self._store.upsert_scim_group_mapping(
            group_id=group_id, external_id=external_id
        )


__all__ = ["ScimDAL"]
