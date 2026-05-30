"""Data-plane authentication helpers.

The sampled Onyx version used PyJWT directly and read DATA_PLANE_SECRET from
an Onyx config var. This repo uses its own HS256 JWT helpers from src/auth
and reads the secret from the standard AGENTIC_SEARCH_AUTH_SECRET env var.

generate_data_plane_token() produces a short-lived HS256 token for
server-to-server calls. control_plane_dep() is a FastAPI dependency that
validates an X-API-KEY header against CONTROL_PLANE_API_KEY env var.
"""

from __future__ import annotations

import logging
import os

from fastapi import HTTPException
from fastapi import Request

from src.backend.auth.users import generate_user_jwt_token
from src.backend.auth.users import user_from_jwt_token

logger = logging.getLogger(__name__)

_CONTROL_PLANE_API_KEY_ENV = "CONTROL_PLANE_API_KEY"
_DATA_PLANE_USER_ID = "data_plane_service"
_TOKEN_EXPIRY_SECONDS = 300  # 5 minutes


def generate_data_plane_token(secret: str | None = None) -> str:
    """Return a short-lived HS256 token for data-plane → control-plane calls."""
    return generate_user_jwt_token(
        user_id=_DATA_PLANE_USER_ID,
        expires_in_seconds=_TOKEN_EXPIRY_SECONDS,
        secret=secret,
    )


def verify_data_plane_token(token: str, secret: str | None = None) -> bool:
    """Return True if *token* is a valid non-expired data-plane token."""
    try:
        payload = user_from_jwt_token(token, secret=secret)
        return payload.id == _DATA_PLANE_USER_ID
    except ValueError:
        return False


async def control_plane_dep(request: Request) -> None:
    """FastAPI dependency: validates X-API-KEY against CONTROL_PLANE_API_KEY env var.

    Returns normally on success; raises HTTP 401/403 on failure.
    In a self-hosted single-tenant deployment this guard is only relevant if
    the instance is used as a data plane — most deployments can leave
    CONTROL_PLANE_API_KEY unset, which disables the guard entirely.
    """
    expected_key = os.environ.get(_CONTROL_PLANE_API_KEY_ENV)
    if not expected_key:
        # No key configured → control-plane endpoints are disabled.
        raise HTTPException(
            status_code=403,
            detail="Control plane API key not configured on this instance.",
        )

    api_key = request.headers.get("X-API-KEY")
    if api_key != expected_key:
        logger.warning("Control plane request rejected: invalid X-API-KEY header.")
        raise HTTPException(status_code=401, detail="Invalid API key.")

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        logger.warning("Control plane request rejected: missing Bearer token.")
        raise HTTPException(status_code=401, detail="Bearer token required.")

    token = auth_header.removeprefix("Bearer ").strip()
    if not verify_data_plane_token(token):
        logger.warning("Control plane request rejected: invalid data-plane token.")
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


__all__ = [
    "control_plane_dep",
    "generate_data_plane_token",
    "verify_data_plane_token",
]
