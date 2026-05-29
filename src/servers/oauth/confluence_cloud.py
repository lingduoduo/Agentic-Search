"""Confluence Cloud OAuth helper.

Adapted from the sampled Onyx ee/onyx/server/oauth/confluence_cloud.py.
Credential DB (SQLAlchemy) callbacks are stubbed with 501 — this deployment
has no connector credential store.  URL generation is unchanged.
"""

from __future__ import annotations

import os

from pydantic import BaseModel


class ConfluenceCloudOAuth:
    CLIENT_ID: str | None = os.environ.get("OAUTH_CONFLUENCE_CLOUD_CLIENT_ID")
    CLIENT_SECRET: str | None = os.environ.get("OAUTH_CONFLUENCE_CLOUD_CLIENT_SECRET")
    TOKEN_URL = "https://auth.atlassian.com/oauth/token"
    ACCESSIBLE_RESOURCE_URL = (
        "https://api.atlassian.com/oauth/token/accessible-resources"
    )

    CONFLUENCE_OAUTH_SCOPE = (
        "read:confluence-space.summary%20"
        "read:confluence-props%20"
        "read:confluence-content.all%20"
        "read:confluence-content.summary%20"
        "read:confluence-content.permission%20"
        "read:confluence-user%20"
        "read:confluence-groups%20"
        "read:space:confluence%20"
        "readonly:content.attachment:confluence%20"
        "search:confluence%20"
        "read:attachment:confluence%20"
        "read:content-details:confluence%20"
        "offline_access"
    )

    class OAuthSession(BaseModel):
        email: str
        redirect_on_success: str | None

    @classmethod
    def _redirect_uri(cls, web_domain: str, dev_mode: bool) -> str:
        base = f"{web_domain}/admin/connectors/confluence/oauth/callback"
        return f"https://redirectmeto.com/{base}" if dev_mode else base

    @classmethod
    def generate_oauth_url(
        cls, state: str, web_domain: str, dev_mode: bool = False
    ) -> str:
        redirect_uri = cls._redirect_uri(web_domain, dev_mode)
        return (
            "https://auth.atlassian.com/authorize"
            f"?audience=api.atlassian.com"
            f"&client_id={cls.CLIENT_ID}"
            f"&scope={cls.CONFLUENCE_OAUTH_SCOPE}"
            f"&redirect_uri={redirect_uri}"
            f"&state={state}"
            "&response_type=code"
            "&prompt=consent"
        )

    @classmethod
    def session_dump_json(cls, email: str, redirect_on_success: str | None) -> str:
        return cls.OAuthSession(
            email=email, redirect_on_success=redirect_on_success
        ).model_dump_json()

    @classmethod
    def parse_session(cls, session_json: str) -> "ConfluenceCloudOAuth.OAuthSession":
        return cls.OAuthSession.model_validate_json(session_json)


__all__ = ["ConfluenceCloudOAuth"]
