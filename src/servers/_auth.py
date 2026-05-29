"""Shared FastAPI dependency factory for admin-only endpoints."""

from __future__ import annotations

from fastapi import HTTPException, Request

from src.auth import AuthenticatedUser, user_from_headers
from src.configs import AppSettings


def make_require_admin(app_settings: AppSettings):
    """Return a FastAPI dependency that enforces super-user access."""

    def _require_admin(request: Request) -> AuthenticatedUser:
        user = user_from_headers(request.headers)
        if user is None or user.is_anonymous:
            raise HTTPException(status_code=401, detail="Authentication required.")
        super_users = app_settings.auth.super_users
        if user.id not in super_users and (
            user.email is None or user.email not in super_users
        ):
            raise HTTPException(status_code=403, detail="Admin access required.")
        return user

    return _require_admin


__all__ = ["make_require_admin"]
