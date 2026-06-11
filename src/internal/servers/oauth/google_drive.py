"""Google Drive OAuth provider."""

from __future__ import annotations

import os

from .provider import ConnectorOAuthProvider

_GOOGLE_OAUTH_SCOPE = " ".join(
    [
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/admin.directory.user.readonly",
        "https://www.googleapis.com/auth/admin.directory.group.readonly",
        "openid",
        "email",
    ]
)


class GoogleDriveOAuth(ConnectorOAuthProvider):
    PROVIDER_ID = "google_drive"
    DISPLAY_NAME = "Google Drive"
    CLIENT_ID_ENV_VAR = "OAUTH_GOOGLE_DRIVE_CLIENT_ID"
    CLIENT_ID: str | None = os.environ.get("OAUTH_GOOGLE_DRIVE_CLIENT_ID")
    CLIENT_SECRET: str | None = os.environ.get("OAUTH_GOOGLE_DRIVE_CLIENT_SECRET")
    AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    CALLBACK_SLUG = "google-drive"
    TOKEN_URL = "https://oauth2.googleapis.com/token"

    @classmethod
    def authorization_params(cls, state: str, redirect_uri: str) -> dict[str, str]:
        return {
            "client_id": cls.CLIENT_ID or "",
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": _GOOGLE_OAUTH_SCOPE,
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }


__all__ = ["GoogleDriveOAuth"]
