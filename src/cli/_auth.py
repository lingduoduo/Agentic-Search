# src/cli/_auth.py
from __future__ import annotations

from src.backend.auth import generate_user_jwt_token


def resolve_token(
    token: str | None,
    user_id: str | None,
    email: str | None = None,
    secret: str | None = None,
) -> str:
    """Return a Bearer JWT.

    Priority: pre-baked ``token`` > mint from ``user_id``.
    Raises ``ValueError`` if neither is supplied.
    """
    if token:
        return token
    if user_id:
        return generate_user_jwt_token(user_id=user_id, email=email, secret=secret)
    raise ValueError("Provide --token or --user-id to authenticate.")
