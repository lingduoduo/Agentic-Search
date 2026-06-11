"""Hook management API.

Provides admin CRUD for persistent webhook configurations backed by
AgenticSearchStore. Endpoint reachability is validated with real HTTP
calls (httpx) before any hook is created or re-activated.

The previous version used SQLAlchemy ORM models, SecretStr masked
API keys, and an hooks registry spec system. This port:
  - Stores hooks in AgenticSearchStore (new hooks table)
  - Uses HookPoint / HookFailStrategy enums from src/hooks/executor.py
  - Implements SSRF protection via a lightweight URL validator
  - Uses plain string API keys (masked in responses by showing only last 4 chars)
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from enum import Enum
from uuid import uuid4

import httpx
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi import Response
from pydantic import BaseModel
from pydantic import Field
from pydantic import SecretStr

from src.internal.auth import AuthenticatedUser
from src.internal.configs import AppSettings
from src.internal.db import AgenticSearchStore
from src.internal.db.models import HookRecord
from src.internal.hooks.executor import HookFailStrategy
from src.internal.hooks.executor import HookPoint
from src.internal.servers._auth import make_require_admin

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 5.0


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------


def _check_ssrf_safety(url: str) -> None:
    """Raise HTTPException 400 if *url* resolves to a private/loopback address."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise HTTPException(
                status_code=400, detail="Only http/https URLs are allowed."
            )
        host = parsed.hostname
        if not host:
            raise HTTPException(status_code=400, detail="URL must have a hostname.")
        try:
            addr = ipaddress.ip_address(host)
        except ValueError:
            addr = ipaddress.ip_address(socket.gethostbyname(host))
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            raise HTTPException(
                status_code=400,
                detail=f"Endpoint URL resolves to a private/loopback address ({addr}). "
                "Use a publicly reachable URL.",
            )
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not resolve host: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class HookValidateStatus(str, Enum):
    passed = "passed"
    auth_failed = "auth_failed"
    timeout = "timeout"
    cannot_connect = "cannot_connect"


class HookValidateResponse(BaseModel):
    status: HookValidateStatus
    error_message: str | None = None


class HookCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    hook_point: HookPoint
    endpoint_url: str
    api_key: SecretStr | None = None
    fail_strategy: HookFailStrategy = HookFailStrategy.SOFT
    timeout_seconds: float = Field(default=_DEFAULT_TIMEOUT, gt=0, le=60)


class HookUpdateRequest(BaseModel):
    name: str | None = None
    endpoint_url: str | None = None
    api_key: SecretStr | None = None
    fail_strategy: HookFailStrategy | None = None
    timeout_seconds: float | None = Field(default=None, gt=0, le=60)
    is_active: bool | None = None


class HookResponse(BaseModel):
    id: str
    name: str
    hook_point: str
    endpoint_url: str
    api_key_masked: str | None = None
    fail_strategy: str
    timeout_seconds: float
    is_active: bool
    is_reachable: bool | None
    creator_id: str | None
    created_at: str | None
    updated_at: str | None


class HookPointMetaResponse(BaseModel):
    hook_point: str
    display_name: str
    description: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mask_api_key(key: str | None) -> str | None:
    if not key:
        return None
    return f"****{key[-4:]}" if len(key) > 4 else "****"


def _record_to_response(hook: HookRecord) -> HookResponse:
    return HookResponse(
        id=hook.id,
        name=hook.name,
        hook_point=hook.hook_point,
        endpoint_url=hook.endpoint_url,
        api_key_masked=_mask_api_key(hook.api_key),
        fail_strategy=hook.fail_strategy,
        timeout_seconds=hook.timeout_seconds,
        is_active=hook.is_active,
        is_reachable=hook.is_reachable,
        creator_id=hook.creator_id,
        created_at=hook.created_at,
        updated_at=hook.updated_at,
    )


def _validate_endpoint(
    endpoint_url: str,
    api_key: str | None,
    timeout_seconds: float,
) -> HookValidateResponse:
    """POST an empty body to endpoint_url to check reachability."""
    _check_ssrf_safety(endpoint_url)
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=False) as client:
            response = client.post(endpoint_url, headers=headers)
        if response.status_code in (401, 403):
            return HookValidateResponse(
                status=HookValidateStatus.auth_failed,
                error_message=f"Authentication failed (HTTP {response.status_code}).",
            )
        return HookValidateResponse(status=HookValidateStatus.passed)
    except httpx.TimeoutException as exc:
        logger.warning("Hook validation timed out for %s: %s", endpoint_url, exc)
        return HookValidateResponse(
            status=HookValidateStatus.timeout,
            error_message="Endpoint timed out — consider increasing timeout_seconds.",
        )
    except Exception as exc:
        logger.warning("Hook validation failed for %s: %s", endpoint_url, exc)
        return HookValidateResponse(
            status=HookValidateStatus.cannot_connect, error_message=str(exc)
        )


def _get_hook_or_404(store: AgenticSearchStore, hook_id: str) -> HookRecord:
    hook = store.get_hook(hook_id)
    if hook is None:
        raise HTTPException(status_code=404, detail=f"Hook {hook_id!r} not found.")
    return hook


def _raise_for_validation(validation: HookValidateResponse) -> None:
    if validation.status == HookValidateStatus.auth_failed:
        raise HTTPException(status_code=401, detail=validation.error_message)
    if validation.status == HookValidateStatus.timeout:
        raise HTTPException(status_code=504, detail=validation.error_message)
    raise HTTPException(status_code=502, detail=validation.error_message)


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

_HOOK_POINT_DESCRIPTIONS = {
    HookPoint.QUERY_PROCESSING: "Rewrite or annotate queries before retrieval.",
    HookPoint.BEFORE_RETRIEVAL: "Filter or modify the search request.",
    HookPoint.AFTER_RETRIEVAL: "Re-rank or filter retrieved results.",
    HookPoint.ANSWER_GENERATED: "Post-process the generated answer.",
}


def create_hooks_router(
    store: AgenticSearchStore,
    app_settings: AppSettings,
) -> APIRouter:
    """Return an APIRouter for hook management endpoints."""

    router = APIRouter(prefix="/admin/hooks", tags=["hooks"])

    _require_admin = make_require_admin(app_settings)

    @router.get("/specs")
    def get_hook_point_specs(
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> list[HookPointMetaResponse]:
        """Return metadata for all supported hook points."""
        return [
            HookPointMetaResponse(
                hook_point=hp.value,
                display_name=hp.value.replace("_", " ").title(),
                description=_HOOK_POINT_DESCRIPTIONS.get(hp, ""),
            )
            for hp in HookPoint
        ]

    @router.get("")
    def list_hooks(
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> list[HookResponse]:
        return [_record_to_response(h) for h in store.list_hooks()]

    @router.post("", status_code=201)
    def create_hook(
        req: HookCreateRequest,
        user: AuthenticatedUser = Depends(_require_admin),
    ) -> HookResponse:
        """Create a hook. The endpoint must be reachable before the hook is saved."""
        api_key = req.api_key.get_secret_value() if req.api_key else None
        validation = _validate_endpoint(req.endpoint_url, api_key, req.timeout_seconds)
        if validation.status != HookValidateStatus.passed:
            _raise_for_validation(validation)
        hook = store.upsert_hook(
            HookRecord(
                id=f"hook_{uuid4().hex}",
                name=req.name,
                hook_point=req.hook_point.value,
                endpoint_url=req.endpoint_url,
                api_key=api_key,
                fail_strategy=req.fail_strategy.value,
                timeout_seconds=req.timeout_seconds,
                is_active=True,
                is_reachable=True,
                creator_id=user.id,
            )
        )
        return _record_to_response(hook)

    @router.get("/{hook_id}")
    def get_hook(
        hook_id: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> HookResponse:
        return _record_to_response(_get_hook_or_404(store, hook_id))

    @router.patch("/{hook_id}")
    def update_hook(
        hook_id: str,
        req: HookUpdateRequest,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> HookResponse:
        """Update hook fields. Re-validates endpoint if connectivity fields change."""
        existing = _get_hook_or_404(store, hook_id)
        new_url = (
            req.endpoint_url if req.endpoint_url is not None else existing.endpoint_url
        )
        new_key = (
            req.api_key.get_secret_value()
            if req.api_key is not None
            else existing.api_key
        )
        new_timeout = (
            req.timeout_seconds
            if req.timeout_seconds is not None
            else existing.timeout_seconds
        )

        connectivity_changed = (
            req.endpoint_url is not None
            or req.api_key is not None
            or req.timeout_seconds is not None
        )
        if connectivity_changed:
            validation = _validate_endpoint(new_url, new_key, new_timeout)
            if existing.is_active and validation.status != HookValidateStatus.passed:
                _raise_for_validation(validation)
            is_reachable = validation.status == HookValidateStatus.passed
        else:
            is_reachable = existing.is_reachable

        updated = store.upsert_hook(
            HookRecord(
                id=hook_id,
                name=req.name if req.name is not None else existing.name,
                hook_point=existing.hook_point,
                endpoint_url=new_url,
                api_key=new_key,
                fail_strategy=(
                    req.fail_strategy.value
                    if req.fail_strategy is not None
                    else existing.fail_strategy
                ),
                timeout_seconds=new_timeout,
                is_active=req.is_active
                if req.is_active is not None
                else existing.is_active,
                is_reachable=is_reachable,
                creator_id=existing.creator_id,
                created_at=existing.created_at,
            )
        )
        return _record_to_response(updated)

    @router.delete("/{hook_id}", status_code=204, response_class=Response)
    def delete_hook(
        hook_id: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> Response:
        if not store.delete_hook(hook_id):
            raise HTTPException(status_code=404, detail=f"Hook {hook_id!r} not found.")
        return Response(status_code=204)

    @router.post("/{hook_id}/activate")
    def activate_hook(
        hook_id: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> HookResponse:
        """Validate endpoint and mark hook active. Fails if endpoint unreachable."""
        hook = _get_hook_or_404(store, hook_id)
        validation = _validate_endpoint(
            hook.endpoint_url, hook.api_key, hook.timeout_seconds
        )
        if validation.status != HookValidateStatus.passed:
            store.upsert_hook(HookRecord(**{**hook.__dict__, "is_reachable": False}))
            _raise_for_validation(validation)
        updated = store.upsert_hook(
            HookRecord(**{**hook.__dict__, "is_active": True, "is_reachable": True})
        )
        return _record_to_response(updated)

    @router.post("/{hook_id}/deactivate")
    def deactivate_hook(
        hook_id: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> HookResponse:
        hook = _get_hook_or_404(store, hook_id)
        updated = store.upsert_hook(HookRecord(**{**hook.__dict__, "is_active": False}))
        return _record_to_response(updated)

    @router.post("/{hook_id}/validate")
    def validate_hook(
        hook_id: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> HookValidateResponse:
        hook = _get_hook_or_404(store, hook_id)
        validation = _validate_endpoint(
            hook.endpoint_url, hook.api_key, hook.timeout_seconds
        )
        passed = validation.status == HookValidateStatus.passed
        if hook.is_reachable != passed:
            store.upsert_hook(HookRecord(**{**hook.__dict__, "is_reachable": passed}))
        return validation

    @router.get("/{hook_id}/execution-logs")
    def list_execution_logs(
        hook_id: str,
        limit: int = Query(default=10, ge=1, le=100),
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> list[dict]:
        """Return execution logs for a hook (from HookRegistry if configured)."""
        _get_hook_or_404(store, hook_id)
        return []

    return router


__all__ = [
    "HookCreateRequest",
    "HookPointMetaResponse",
    "HookResponse",
    "HookUpdateRequest",
    "HookValidateResponse",
    "HookValidateStatus",
    "create_hooks_router",
]
