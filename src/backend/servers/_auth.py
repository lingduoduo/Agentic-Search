"""Shared FastAPI dependency factory for admin-only endpoints."""

from __future__ import annotations

from fastapi import HTTPException, Request

from src.backend.auth import AuthenticatedUser, user_from_headers
from src.backend.configs import AppSettings


def make_require_admin(app_settings: AppSettings):
    """Return a FastAPI dependency that enforces super-user access.

    A user is considered admin if they appear in the configured super_users
    list OR if their JWT token carries role="admin" (set at login time).
    """

    def _require_admin(request: Request) -> AuthenticatedUser:
        user = user_from_headers(request.headers)
        if user is None or user.is_anonymous:
            raise HTTPException(status_code=401, detail="Authentication required.")
        super_users = app_settings.auth.super_users
        in_super_users = user.id in super_users or (
            user.email is not None and user.email in super_users
        )
        has_admin_role = user.metadata.get("role") == "admin"
        if not in_super_users and not has_admin_role:
            raise HTTPException(status_code=403, detail="Admin access required.")
        return user

    return _require_admin


__all__ = ["make_require_admin"]
