"""Slack OAuth provider."""

from __future__ import annotations

import os

from .provider import ConnectorOAuthProvider


class SlackOAuth(ConnectorOAuthProvider):
    PROVIDER_ID = "slack"
    DISPLAY_NAME = "Slack"
    CLIENT_ID_ENV_VAR = "OAUTH_SLACK_CLIENT_ID"
    CLIENT_ID: str | None = os.environ.get("OAUTH_SLACK_CLIENT_ID")
    CLIENT_SECRET: str | None = os.environ.get("OAUTH_SLACK_CLIENT_SECRET")
    AUTHORIZATION_URL = "https://slack.com/oauth/v2/authorize"
    CALLBACK_SLUG = "slack"
    TOKEN_URL = "https://slack.com/api/oauth.v2.access"

    SLACK_OAUTH_SCOPE = "channels:history,channels:join,channels:read,groups:history,groups:read,im:history,im:read,mpim:history,mpim:read,users:read,users:read.email,team:read"

    @classmethod
    def authorization_params(cls, state: str, redirect_uri: str) -> dict[str, str]:
        return {
            "client_id": cls.CLIENT_ID or "",
            "scope": cls.SLACK_OAUTH_SCOPE,
            "redirect_uri": redirect_uri,
            "state": state,
        }


__all__ = ["SlackOAuth"]
