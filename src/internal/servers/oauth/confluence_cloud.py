"""Confluence Cloud OAuth provider."""

from __future__ import annotations

import os

from .provider import ConnectorOAuthProvider


class ConfluenceCloudOAuth(ConnectorOAuthProvider):
    PROVIDER_ID = "confluence"
    DISPLAY_NAME = "Confluence"
    CLIENT_ID_ENV_VAR = "OAUTH_CONFLUENCE_CLOUD_CLIENT_ID"
    CLIENT_ID: str | None = os.environ.get("OAUTH_CONFLUENCE_CLOUD_CLIENT_ID")
    CLIENT_SECRET: str | None = os.environ.get("OAUTH_CONFLUENCE_CLOUD_CLIENT_SECRET")
    AUTHORIZATION_URL = "https://auth.atlassian.com/authorize"
    CALLBACK_SLUG = "confluence"
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

    @classmethod
    def authorization_params(cls, state: str, redirect_uri: str) -> dict[str, str]:
        return {
            "audience": "api.atlassian.com",
            "client_id": cls.CLIENT_ID or "",
            "scope": cls.CONFLUENCE_OAUTH_SCOPE.replace("%20", " "),
            "redirect_uri": redirect_uri,
            "state": state,
            "response_type": "code",
            "prompt": "consent",
        }


__all__ = ["ConfluenceCloudOAuth"]
