"""SCIM bearer token authentication.

Adapted from the sampled Onyx ee/onyx/server/scim/auth.py.
ScimDAL backed by AgenticSearchStore instead of SQLAlchemy.
get_hashed_bearer_token_from_request replaced with a local extraction helper.
"""

from __future__ import annotations

import hashlib
import secrets

from fastapi import Request

from src.backend.db import AgenticSearchStore
from src.backend.servers.scim.dal import ScimDAL


class ScimAuthError(Exception):
    """Raised when SCIM bearer token authentication fails."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


SCIM_TOKEN_PREFIX = "agentic_search_scim_"
SCIM_TOKEN_LENGTH = 48


def _hash_scim_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_scim_token() -> tuple[str, str, str]:
    """Generate a new SCIM bearer token.

    Returns:
        ``(raw_token, hashed_token, token_display)``
    """
    raw_token = SCIM_TOKEN_PREFIX + secrets.token_urlsafe(SCIM_TOKEN_LENGTH)
    hashed_token = _hash_scim_token(raw_token)
    token_display = SCIM_TOKEN_PREFIX + "****" + raw_token[-4:]
    return raw_token, hashed_token, token_display


def _extract_bearer_token(request: Request) -> str | None:
    """Extract a bearer token from the Authorization header."""
    auth = request.headers.get("Authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    return auth[7:].strip() or None


def _get_hashed_scim_token_from_request(request: Request) -> str | None:
    """Extract and hash a SCIM token from the Authorization header."""
    raw = _extract_bearer_token(request)
    if not raw or not raw.startswith(SCIM_TOKEN_PREFIX):
        return None
    return _hash_scim_token(raw)


def make_verify_scim_token(store: AgenticSearchStore):
    """Return a FastAPI dependency that verifies SCIM bearer tokens."""
    dal = ScimDAL(store)

    def verify_scim_token(request: Request) -> dict:
        hashed = _get_hashed_scim_token_from_request(request)
        if not hashed:
            raise ScimAuthError(401, "Missing or invalid SCIM bearer token")
        token = dal.get_token_by_hash(hashed)
        if not token:
            raise ScimAuthError(401, "Invalid SCIM bearer token")
        if not token["is_active"]:
            raise ScimAuthError(401, "SCIM token has been revoked")
        dal.update_token_last_used(str(token["id"]))
        return token

    return verify_scim_token


__all__ = [
    "ScimAuthError",
    "generate_scim_token",
    "make_verify_scim_token",
]
