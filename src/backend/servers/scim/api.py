"""SCIM 2.0 API endpoints (RFC 7644).

Adapted from the sampled Onyx ee/onyx/server/scim/api.py.
Changes:
- ScimDAL backed by AgenticSearchStore (no SQLAlchemy).
- ORM User/UserGroup replaced with ScimUser/ScimGroup dataclasses.
- Seat-limit checking removed (no license seat enforcement in this deployment).
- Permission recomputation removed (no permission graph in this deployment).
- Token verification uses make_verify_scim_token factory with Depends.
- register_scim_exception_handlers wires ScimAuthError as HTTP 401.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter
from fastapi import Depends
from fastapi import FastAPI
from fastapi import Query
from fastapi import Request
from fastapi import Response
from fastapi.responses import JSONResponse

from src.backend.db import AgenticSearchStore
from src.backend.servers.scim.auth import ScimAuthError
from src.backend.servers.scim.auth import generate_scim_token
from src.backend.servers.scim.auth import make_verify_scim_token
from src.backend.servers.scim.dal import ScimDAL
from src.backend.servers.scim.filtering import parse_scim_filter
from src.backend.servers.scim.models import SCIM_LIST_RESPONSE_SCHEMA
from src.backend.servers.scim.models import ScimError
from src.backend.servers.scim.models import ScimGroupMember
from src.backend.servers.scim.models import ScimGroupResource
from src.backend.servers.scim.models import ScimListResponse
from src.backend.servers.scim.models import ScimMappingFields
from src.backend.servers.scim.models import ScimPatchRequest
from src.backend.servers.scim.models import ScimServiceProviderConfig
from src.backend.servers.scim.models import ScimTokenCreate
from src.backend.servers.scim.models import ScimTokenCreatedResponse
from src.backend.servers.scim.models import ScimTokenResponse
from src.backend.servers.scim.models import ScimUserResource
from src.backend.servers.scim.patch import ScimPatchError
from src.backend.servers.scim.patch import apply_group_patch
from src.backend.servers.scim.patch import apply_user_patch
from src.backend.servers.scim.providers.base import ScimUser
from src.backend.servers.scim.providers.base import get_default_provider
from src.backend.servers.scim.providers.base import serialize_emails
from src.backend.servers.scim.schema_definitions import ENTERPRISE_USER_SCHEMA_DEF
from src.backend.servers.scim.schema_definitions import GROUP_RESOURCE_TYPE
from src.backend.servers.scim.schema_definitions import GROUP_SCHEMA_DEF
from src.backend.servers.scim.schema_definitions import SERVICE_PROVIDER_CONFIG
from src.backend.servers.scim.schema_definitions import USER_RESOURCE_TYPE
from src.backend.servers.scim.schema_definitions import USER_SCHEMA_DEF

logger = logging.getLogger(__name__)

_RESERVED_GROUP_NAMES: frozenset[str] = frozenset({"Admin", "Basic"})


class ScimJSONResponse(JSONResponse):
    media_type = "application/scim+json"


def register_scim_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ScimAuthError)
    async def _handle_scim_auth_error(
        _request: Request, exc: ScimAuthError
    ) -> ScimJSONResponse:
        return _scim_error_response(exc.status_code, exc.detail)


def _scim_error_response(status: int, detail: str) -> ScimJSONResponse:
    logger.warning("SCIM error: status=%s detail=%s", status, detail)
    body = ScimError(status=str(status), detail=detail)
    return ScimJSONResponse(
        status_code=status, content=body.model_dump(exclude_none=True)
    )


def _parse_excluded_attributes(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {attr.strip().lower() for attr in raw.split(",") if attr.strip()}


def _apply_exclusions(
    resource: ScimUserResource | ScimGroupResource, excluded: set[str]
) -> dict:
    data = resource.model_dump(exclude_none=True, by_alias=True)
    for attr in excluded:
        for k in [k for k in data if k.lower() == attr]:
            del data[k]
    return data


def _scim_resource_response(
    resource: ScimUserResource | ScimGroupResource | ScimListResponse,
    status_code: int = 200,
) -> ScimJSONResponse:
    content = resource.model_dump(exclude_none=True, by_alias=True)
    return ScimJSONResponse(status_code=status_code, content=content)


def _build_list_response(
    resources: list,
    total: int,
    start_index: int,
    count: int,
    excluded: set[str] | None = None,
) -> ScimListResponse | ScimJSONResponse:
    if excluded:
        envelope = ScimListResponse(
            totalResults=total, startIndex=start_index, itemsPerPage=count
        )
        data = envelope.model_dump(exclude_none=True)
        data["Resources"] = [_apply_exclusions(r, excluded) for r in resources]
        return ScimJSONResponse(content=data)
    return _scim_resource_response(
        ScimListResponse(
            totalResults=total,
            startIndex=start_index,
            itemsPerPage=count,
            Resources=resources,
        )
    )


def _mapping_to_fields(mapping: dict | None) -> ScimMappingFields | None:
    if not mapping:
        return None
    return ScimMappingFields(
        department=mapping.get("department"),
        manager=mapping.get("manager"),
        given_name=mapping.get("given_name"),
        family_name=mapping.get("family_name"),
        scim_emails_json=mapping.get("scim_emails_json"),
    )


def _fields_from_resource(resource: ScimUserResource) -> ScimMappingFields:
    ext = resource.enterprise_extension
    department = ext.department if ext else None
    manager = ext.manager.value if ext and ext.manager else None
    return ScimMappingFields(
        department=department,
        manager=manager,
        given_name=resource.name.givenName if resource.name else None,
        family_name=resource.name.familyName if resource.name else None,
        scim_emails_json=serialize_emails(resource.emails),
    )


def _scim_name_str(resource: ScimUserResource) -> str | None:
    name = resource.name
    if not name:
        return None
    if name.formatted:
        return name.formatted
    parts = " ".join(p for p in [name.givenName, name.familyName] if p)
    return parts or None


def _is_valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


def _parse_members(
    members: list[ScimGroupMember], dal: ScimDAL
) -> tuple[list[str], str | None]:
    uuids: list[str] = []
    for m in members:
        if not _is_valid_uuid(m.value):
            return [], f"Invalid member ID: {m.value}"
        uuids.append(m.value)
    if uuids:
        missing = dal.validate_member_ids(uuids)
        if missing:
            return [], f"Member(s) not found: {', '.join(missing)}"
    return uuids, None


def create_scim_router(store: AgenticSearchStore) -> APIRouter:
    """Return a SCIM 2.0 APIRouter bound to *store*."""

    scim_router = APIRouter(prefix="/scim/v2", tags=["SCIM"])
    _verify_token = make_verify_scim_token(store)
    dal = ScimDAL(store)

    def _auth(request: Request) -> dict:
        return _verify_token(request)

    provider = get_default_provider()

    # ---------------------------------------------------------------------------
    # Service Discovery (unauthenticated)
    # ---------------------------------------------------------------------------

    @scim_router.get("/ServiceProviderConfig")
    def get_service_provider_config() -> ScimServiceProviderConfig:
        return SERVICE_PROVIDER_CONFIG

    @scim_router.get("/ResourceTypes")
    def get_resource_types() -> ScimJSONResponse:
        resources = [USER_RESOURCE_TYPE, GROUP_RESOURCE_TYPE]
        return ScimJSONResponse(
            content={
                "schemas": [SCIM_LIST_RESPONSE_SCHEMA],
                "totalResults": len(resources),
                "Resources": [
                    r.model_dump(exclude_none=True, by_alias=True) for r in resources
                ],
            }
        )

    @scim_router.get("/Schemas")
    def get_schemas() -> ScimJSONResponse:
        schemas = [USER_SCHEMA_DEF, GROUP_SCHEMA_DEF, ENTERPRISE_USER_SCHEMA_DEF]
        return ScimJSONResponse(
            content={
                "schemas": [SCIM_LIST_RESPONSE_SCHEMA],
                "totalResults": len(schemas),
                "Resources": [s.model_dump(exclude_none=True) for s in schemas],
            }
        )

    # ---------------------------------------------------------------------------
    # User CRUD
    # ---------------------------------------------------------------------------

    @scim_router.get("/Users", response_model=None)
    def list_users(
        filter: str | None = Query(None),
        excludedAttributes: str | None = None,
        startIndex: int = Query(1, ge=1),
        count: int = Query(100, ge=0, le=500),
        _token: dict = Depends(_auth),
    ) -> ScimListResponse | ScimJSONResponse:
        try:
            scim_filter = parse_scim_filter(filter)
        except ValueError as e:
            return _scim_error_response(400, str(e))
        try:
            users_with_mappings, total = dal.list_users(scim_filter, startIndex, count)
        except ValueError as e:
            return _scim_error_response(400, str(e))

        user_groups_map = dal.get_users_groups_batch(
            [u.id for u, _ in users_with_mappings]
        )
        resources = [
            provider.build_user_resource(
                user,
                mapping["external_id"] if mapping else None,
                groups=user_groups_map.get(user.id, []),
                scim_username=mapping["scim_username"] if mapping else None,
                fields=_mapping_to_fields(mapping),
            )
            for user, mapping in users_with_mappings
        ]
        return _build_list_response(
            resources,
            total,
            startIndex,
            count,
            excluded=_parse_excluded_attributes(excludedAttributes),
        )

    @scim_router.get("/Users/{user_id}", response_model=None)
    def get_user(
        user_id: str,
        excludedAttributes: str | None = None,
        _token: dict = Depends(_auth),
    ) -> ScimUserResource | ScimJSONResponse:
        user = dal.get_user(user_id)
        if not user:
            return _scim_error_response(404, f"User {user_id} not found")
        mapping = dal.get_user_mapping(user.id)
        resource = provider.build_user_resource(
            user,
            mapping["external_id"] if mapping else None,
            groups=dal.get_user_groups(user.id),
            scim_username=mapping["scim_username"] if mapping else None,
            fields=_mapping_to_fields(mapping),
        )
        excluded = _parse_excluded_attributes(excludedAttributes)
        if excluded:
            return ScimJSONResponse(content=_apply_exclusions(resource, excluded))
        return _scim_resource_response(resource)

    @scim_router.post("/Users", status_code=201, response_model=None)
    def create_user(
        user_resource: ScimUserResource,
        _token: dict = Depends(_auth),
    ) -> ScimUserResource | ScimJSONResponse:
        email = user_resource.userName.strip()
        external_id = user_resource.externalId
        scim_username = user_resource.userName.strip()
        fields = _fields_from_resource(user_resource)

        existing = dal.get_user_by_email(email)
        if existing:
            if dal.get_user_mapping(existing.id):
                return _scim_error_response(
                    409, f"User with email {email} already exists"
                )
            if user_resource.active and not existing.is_active:
                dal.update_user(existing, is_active=True)
            dal.update_user(existing, personal_name=_scim_name_str(user_resource))
            dal.create_user_mapping(
                external_id=external_id,
                user_id=existing.id,
                scim_username=scim_username,
                fields=fields,
            )
            return _scim_resource_response(
                provider.build_user_resource(
                    existing, external_id, scim_username=scim_username, fields=fields
                ),
                status_code=201,
            )

        user = ScimUser(
            id=str(uuid.uuid4()),
            email=email,
            is_active=user_resource.active,
            personal_name=_scim_name_str(user_resource),
        )
        dal.add_user(user)
        dal.create_user_mapping(
            external_id=external_id,
            user_id=user.id,
            scim_username=scim_username,
            fields=fields,
        )
        return _scim_resource_response(
            provider.build_user_resource(
                user, external_id, scim_username=scim_username, fields=fields
            ),
            status_code=201,
        )

    @scim_router.put("/Users/{user_id}", response_model=None)
    def replace_user(
        user_id: str,
        user_resource: ScimUserResource,
        _token: dict = Depends(_auth),
    ) -> ScimUserResource | ScimJSONResponse:
        user = dal.get_user(user_id)
        if not user:
            return _scim_error_response(404, f"User {user_id} not found")
        dal.update_user(
            user,
            email=user_resource.userName.strip(),
            is_active=user_resource.active,
            personal_name=_scim_name_str(user_resource),
        )
        new_ext = user_resource.externalId
        scim_username = user_resource.userName.strip()
        fields = _fields_from_resource(user_resource)
        dal.sync_user_external_id(
            user.id, new_ext, scim_username=scim_username, fields=fields
        )
        return _scim_resource_response(
            provider.build_user_resource(
                user,
                new_ext,
                groups=dal.get_user_groups(user.id),
                scim_username=scim_username,
                fields=fields,
            )
        )

    @scim_router.patch("/Users/{user_id}", response_model=None)
    def patch_user(
        user_id: str,
        patch_request: ScimPatchRequest,
        _token: dict = Depends(_auth),
    ) -> ScimUserResource | ScimJSONResponse:
        user = dal.get_user(user_id)
        if not user:
            return _scim_error_response(404, f"User {user_id} not found")
        mapping = dal.get_user_mapping(user.id)
        ext_id = mapping["external_id"] if mapping else None
        cur_username = mapping["scim_username"] if mapping else None
        cur_fields = _mapping_to_fields(mapping)
        current = provider.build_user_resource(
            user,
            ext_id,
            groups=dal.get_user_groups(user.id),
            scim_username=cur_username,
            fields=cur_fields,
        )
        try:
            patched, ent_data = apply_user_patch(
                patch_request.Operations, current, provider.ignored_patch_paths
            )
        except ScimPatchError as e:
            return _scim_error_response(e.status, e.detail)

        personal_name = (
            patched.displayName
            if patched.displayName and patched.displayName != current.displayName
            else _scim_name_str(patched)
        )
        new_email = patched.userName.strip() if patched.userName else None
        email_changed = new_email and new_email.lower() != user.email.lower()
        dal.update_user(
            user,
            email=new_email if email_changed else None,
            is_active=patched.active if patched.active != user.is_active else None,
            personal_name=personal_name,
        )
        cf = cur_fields or ScimMappingFields()
        fields = ScimMappingFields(
            department=ent_data.get("department", cf.department),
            manager=ent_data.get("manager", cf.manager),
            given_name=patched.name.givenName if patched.name else cf.given_name,
            family_name=patched.name.familyName if patched.name else cf.family_name,
            scim_emails_json=(
                serialize_emails(patched.emails)
                if patched.emails is not None
                else cf.scim_emails_json
            ),
        )
        new_username = patched.userName.strip() if patched.userName else None
        dal.sync_user_external_id(
            user.id, patched.externalId, scim_username=new_username, fields=fields
        )
        return _scim_resource_response(
            provider.build_user_resource(
                user,
                patched.externalId,
                groups=dal.get_user_groups(user.id),
                scim_username=new_username,
                fields=fields,
            )
        )

    @scim_router.delete("/Users/{user_id}", status_code=204, response_model=None)
    def delete_user(
        user_id: str,
        _token: dict = Depends(_auth),
    ) -> Response | ScimJSONResponse:
        user = dal.get_user(user_id)
        if not user:
            return _scim_error_response(404, f"User {user_id} not found")
        if not dal.get_user_mapping(user.id):
            return _scim_error_response(404, f"User {user_id} not found")
        dal.deactivate_user(user)
        dal.delete_user_mapping(user.id)
        return Response(status_code=204)

    # ---------------------------------------------------------------------------
    # Group CRUD
    # ---------------------------------------------------------------------------

    @scim_router.get("/Groups", response_model=None)
    def list_groups(
        filter: str | None = Query(None),
        excludedAttributes: str | None = None,
        startIndex: int = Query(1, ge=1),
        count: int = Query(100, ge=0, le=500),
        _token: dict = Depends(_auth),
    ) -> ScimListResponse | ScimJSONResponse:
        try:
            scim_filter = parse_scim_filter(filter)
        except ValueError as e:
            return _scim_error_response(400, str(e))
        try:
            groups_with_ext_ids, total = dal.list_groups(scim_filter, startIndex, count)
        except ValueError as e:
            return _scim_error_response(400, str(e))
        resources = [
            provider.build_group_resource(
                group, dal.get_group_members(group.id), ext_id
            )
            for group, ext_id in groups_with_ext_ids
        ]
        return _build_list_response(
            resources,
            total,
            startIndex,
            count,
            excluded=_parse_excluded_attributes(excludedAttributes),
        )

    @scim_router.get("/Groups/{group_id}", response_model=None)
    def get_group(
        group_id: str,
        excludedAttributes: str | None = None,
        _token: dict = Depends(_auth),
    ) -> ScimJSONResponse:
        group = dal.get_group(group_id)
        if not group:
            return _scim_error_response(404, f"Group {group_id} not found")
        mapping = dal.get_group_mapping(group.id)
        resource = provider.build_group_resource(
            group,
            dal.get_group_members(group.id),
            mapping["external_id"] if mapping else None,
        )
        excluded = _parse_excluded_attributes(excludedAttributes)
        if excluded:
            return ScimJSONResponse(content=_apply_exclusions(resource, excluded))
        return _scim_resource_response(resource)

    @scim_router.post("/Groups", status_code=201, response_model=None)
    def create_group(
        group_resource: ScimGroupResource,
        _token: dict = Depends(_auth),
    ) -> ScimJSONResponse:
        if group_resource.displayName in _RESERVED_GROUP_NAMES:
            return _scim_error_response(
                409, f"'{group_resource.displayName}' is a reserved group name."
            )
        if dal.get_group_by_name(group_resource.displayName):
            return _scim_error_response(
                409, f"Group '{group_resource.displayName}' already exists"
            )
        member_ids, err = _parse_members(group_resource.members, dal)
        if err:
            return _scim_error_response(400, err)
        group = dal.create_group(group_resource.displayName)
        dal.set_group_members(group.id, member_ids)
        ext_id = group_resource.externalId
        if ext_id:
            dal.create_group_mapping(group.id, ext_id)
        return _scim_resource_response(
            provider.build_group_resource(
                group, dal.get_group_members(group.id), ext_id
            ),
            status_code=201,
        )

    @scim_router.put("/Groups/{group_id}", response_model=None)
    def replace_group(
        group_id: str,
        group_resource: ScimGroupResource,
        _token: dict = Depends(_auth),
    ) -> ScimJSONResponse:
        group = dal.get_group(group_id)
        if not group:
            return _scim_error_response(404, f"Group {group_id} not found")
        if (
            group.name in _RESERVED_GROUP_NAMES
            and group_resource.displayName != group.name
        ):
            return _scim_error_response(
                409, f"'{group.name}' is a reserved group name."
            )
        member_ids, err = _parse_members(group_resource.members, dal)
        if err:
            return _scim_error_response(400, err)
        dal.update_group(group, name=group_resource.displayName)
        dal.set_group_members(group.id, member_ids)
        dal.sync_group_external_id(group.id, group_resource.externalId)
        return _scim_resource_response(
            provider.build_group_resource(
                group, dal.get_group_members(group.id), group_resource.externalId
            )
        )

    @scim_router.patch("/Groups/{group_id}", response_model=None)
    def patch_group(
        group_id: str,
        patch_request: ScimPatchRequest,
        _token: dict = Depends(_auth),
    ) -> ScimJSONResponse:
        group = dal.get_group(group_id)
        if not group:
            return _scim_error_response(404, f"Group {group_id} not found")
        mapping = dal.get_group_mapping(group.id)
        ext_id = mapping["external_id"] if mapping else None
        current = provider.build_group_resource(
            group, dal.get_group_members(group.id), ext_id
        )
        try:
            patched, added_ids, removed_ids = apply_group_patch(
                patch_request.Operations, current, provider.ignored_patch_paths
            )
        except ScimPatchError as e:
            return _scim_error_response(e.status, e.detail)
        new_name = patched.displayName if patched.displayName != group.name else None
        if group.name in _RESERVED_GROUP_NAMES and new_name:
            return _scim_error_response(
                409, f"'{group.name}' is a reserved group name."
            )
        dal.update_group(group, name=new_name)
        if added_ids:
            valid = [mid for mid in added_ids if _is_valid_uuid(mid)]
            missing = dal.validate_member_ids(valid)
            if missing:
                return _scim_error_response(
                    400, f"Member(s) not found: {', '.join(missing)}"
                )
            dal.add_group_members(group.id, valid)
        if removed_ids:
            dal.remove_group_members(
                group.id, [mid for mid in removed_ids if _is_valid_uuid(mid)]
            )
        dal.sync_group_external_id(group.id, patched.externalId)
        return _scim_resource_response(
            provider.build_group_resource(
                group, dal.get_group_members(group.id), patched.externalId
            )
        )

    @scim_router.delete("/Groups/{group_id}", status_code=204, response_model=None)
    def delete_group(
        group_id: str,
        _token: dict = Depends(_auth),
    ) -> Response | ScimJSONResponse:
        group = dal.get_group(group_id)
        if not group:
            return _scim_error_response(404, f"Group {group_id} not found")
        if group.name in _RESERVED_GROUP_NAMES:
            return _scim_error_response(
                409, f"'{group.name}' is a reserved group name."
            )
        dal.delete_group_mapping(group.id)
        dal.delete_group(group)
        return Response(status_code=204)

    # ---------------------------------------------------------------------------
    # Token admin endpoints (Onyx-internal)
    # ---------------------------------------------------------------------------

    @scim_router.post("/tokens", status_code=201)
    def create_token(req: ScimTokenCreate) -> ScimTokenCreatedResponse:
        raw, hashed, display = generate_scim_token()
        token = dal.create_token(
            name=req.name, token_hash=hashed, token_display=display
        )
        return ScimTokenCreatedResponse(
            id=int(hash(token["id"])) & 0x7FFFFFFF,
            name=str(token["name"]),
            token_display=str(token["token_display"]),
            is_active=bool(token["is_active"]),
            created_at=token["created_at"],  # type: ignore[arg-type]
            last_used_at=token.get("last_used_at"),  # type: ignore[arg-type]
            raw_token=raw,
        )

    @scim_router.get("/tokens")
    def list_tokens() -> list[ScimTokenResponse]:
        return [
            ScimTokenResponse(
                id=int(hash(t["id"])) & 0x7FFFFFFF,
                name=str(t["name"]),
                token_display=str(t["token_display"]),
                is_active=bool(t["is_active"]),
                created_at=t["created_at"],  # type: ignore[arg-type]
                last_used_at=t.get("last_used_at"),  # type: ignore[arg-type]
            )
            for t in dal.list_tokens()
        ]

    @scim_router.delete("/tokens/{token_id}", status_code=204)
    def revoke_token(token_id: str) -> Response:
        dal.revoke_token(token_id)
        return Response(status_code=204)

    return scim_router


__all__ = ["create_scim_router", "register_scim_exception_handlers"]
