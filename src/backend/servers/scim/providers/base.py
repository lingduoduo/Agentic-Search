"""Base SCIM provider abstraction.

py.
ORM User/UserGroup replaced with lightweight ScimUser/ScimGroup dataclasses.
"""

from __future__ import annotations

import json
import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

from pydantic import ValidationError

from src.backend.servers.scim.models import SCIM_ENTERPRISE_USER_SCHEMA
from src.backend.servers.scim.models import SCIM_USER_SCHEMA
from src.backend.servers.scim.models import ScimEmail
from src.backend.servers.scim.models import ScimEnterpriseExtension
from src.backend.servers.scim.models import ScimGroupMember
from src.backend.servers.scim.models import ScimGroupResource
from src.backend.servers.scim.models import ScimManagerRef
from src.backend.servers.scim.models import ScimMappingFields
from src.backend.servers.scim.models import ScimMeta
from src.backend.servers.scim.models import ScimName
from src.backend.servers.scim.models import ScimUserGroupRef
from src.backend.servers.scim.models import ScimUserResource

logger = logging.getLogger(__name__)

COMMON_IGNORED_PATCH_PATHS: frozenset[str] = frozenset({"id", "schemas", "meta"})


@dataclass
class ScimUser:
    """Lightweight user record for the SCIM layer (replaces ORM User)."""

    id: str
    email: str
    is_active: bool = True
    personal_name: str | None = None


@dataclass
class ScimGroup:
    """Lightweight group record for the SCIM layer (replaces ORM UserGroup)."""

    id: str
    name: str


class ScimProvider(ABC):
    """Base class for provider-specific SCIM behavior."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def ignored_patch_paths(self) -> frozenset[str]: ...

    @property
    def user_schemas(self) -> list[str]:
        return [SCIM_USER_SCHEMA]

    def build_user_resource(
        self,
        user: ScimUser,
        external_id: str | None = None,
        groups: list[tuple[str, str | None]] | None = None,
        scim_username: str | None = None,
        fields: ScimMappingFields | None = None,
    ) -> ScimUserResource:
        f = fields or ScimMappingFields()
        group_refs = [
            ScimUserGroupRef(value=gid, display=gname) for gid, gname in (groups or [])
        ]
        username = scim_username or user.email
        enterprise_ext: ScimEnterpriseExtension | None = None
        schemas = list(self.user_schemas)
        if f.department is not None or f.manager is not None:
            manager_ref = (
                ScimManagerRef(value=f.manager) if f.manager is not None else None
            )
            enterprise_ext = ScimEnterpriseExtension(
                department=f.department, manager=manager_ref
            )
            if SCIM_ENTERPRISE_USER_SCHEMA not in schemas:
                schemas.append(SCIM_ENTERPRISE_USER_SCHEMA)

        name = self.build_scim_name(user, f)
        emails = _deserialize_emails(f.scim_emails_json, username)

        resource = ScimUserResource(
            schemas=schemas,
            id=user.id,
            externalId=external_id,
            userName=username,
            name=name,
            displayName=user.personal_name,
            emails=emails,
            active=user.is_active,
            groups=group_refs,
            meta=ScimMeta(resourceType="User"),
        )
        resource.enterprise_extension = enterprise_ext
        return resource

    def build_group_resource(
        self,
        group: ScimGroup,
        members: list[tuple[str, str | None]],
        external_id: str | None = None,
    ) -> ScimGroupResource:
        scim_members = [
            ScimGroupMember(value=uid, display=email) for uid, email in members
        ]
        return ScimGroupResource(
            id=group.id,
            externalId=external_id,
            displayName=group.name,
            members=scim_members,
            meta=ScimMeta(resourceType="Group"),
        )

    def build_scim_name(self, user: ScimUser, fields: ScimMappingFields) -> ScimName:
        if fields.given_name is not None or fields.family_name is not None:
            return ScimName(
                givenName=fields.given_name or "",
                familyName=fields.family_name or "",
                formatted=user.personal_name or "",
            )
        if not user.personal_name:
            local = user.email.split("@")[0] if user.email else ""
            return ScimName(givenName=local, familyName="", formatted=local)
        parts = user.personal_name.split(" ", 1)
        return ScimName(
            givenName=parts[0],
            familyName=parts[1] if len(parts) > 1 else "",
            formatted=user.personal_name,
        )


def _deserialize_emails(stored_json: str | None, username: str) -> list[ScimEmail]:
    if stored_json:
        try:
            entries = json.loads(stored_json)
            if isinstance(entries, list) and entries:
                return [ScimEmail(**e) for e in entries]
        except (json.JSONDecodeError, TypeError, ValidationError):
            logger.warning("Corrupt scim_emails_json, falling back: %s", stored_json)
    return [ScimEmail(value=username, type="work", primary=True)]


def serialize_emails(emails: list[ScimEmail]) -> str | None:
    if not emails:
        return None
    return json.dumps([e.model_dump(exclude_none=True) for e in emails])


def get_default_provider() -> ScimProvider:
    from src.backend.servers.scim.providers.okta import OktaProvider

    return OktaProvider()


__all__ = [
    "COMMON_IGNORED_PATCH_PATHS",
    "ScimGroup",
    "ScimProvider",
    "ScimUser",
    "get_default_provider",
    "serialize_emails",
]
