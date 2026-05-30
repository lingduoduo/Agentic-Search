"""Google Drive OAuth helper.

Adapted from the sampled Onyx ee/onyx/server/oauth/google_drive.py.
Credential callbacks stubbed with 501; URL generation is unchanged.
"""

from __future__ import annotations

import os

from pydantic import BaseModel

_GOOGLE_OAUTH_SCOPE = " ".join(
    [
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/admin.directory.user.readonly",
        "https://www.googleapis.com/auth/admin.directory.group.readonly",
        "openid",
        "email",
    ]
)


class GoogleDriveOAuth:
    CLIENT_ID: str | None = os.environ.get("OAUTH_GOOGLE_DRIVE_CLIENT_ID")
    CLIENT_SECRET: str | None = os.environ.get("OAUTH_GOOGLE_DRIVE_CLIENT_SECRET")
    TOKEN_URL = "https://oauth2.googleapis.com/token"

    class OAuthSession(BaseModel):
        email: str
        redirect_on_success: str | None

    @classmethod
    def _redirect_uri(cls, web_domain: str, dev_mode: bool) -> str:
        base = f"{web_domain}/admin/connectors/google-drive/oauth/callback"
        return f"https://redirectmeto.com/{base}" if dev_mode else base

    @classmethod
    def generate_oauth_url(
        cls, state: str, web_domain: str, dev_mode: bool = False
    ) -> str:
        redirect_uri = cls._redirect_uri(web_domain, dev_mode)
        return (
            "https://accounts.google.com/o/oauth2/v2/auth"
            f"?client_id={cls.CLIENT_ID}"
            f"&redirect_uri={redirect_uri}"
            f"&response_type=code"
            f"&scope={_GOOGLE_OAUTH_SCOPE.replace(' ', '%20')}"
            f"&state={state}"
            "&access_type=offline"
            "&prompt=consent"
        )

    @classmethod
    def session_dump_json(cls, email: str, redirect_on_success: str | None) -> str:
        return cls.OAuthSession(
            email=email, redirect_on_success=redirect_on_success
        ).model_dump_json()

    @classmethod
    def parse_session(cls, session_json: str) -> "GoogleDriveOAuth.OAuthSession":
        return cls.OAuthSession.model_validate_json(session_json)


__all__ = ["GoogleDriveOAuth"]
