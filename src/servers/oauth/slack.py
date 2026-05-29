"""Slack OAuth helper.

Adapted from the sampled Onyx ee/onyx/server/oauth/slack.py.
Credential callbacks stubbed with 501; URL generation is unchanged.
"""

from __future__ import annotations

import os

from pydantic import BaseModel


class SlackOAuth:
    CLIENT_ID: str | None = os.environ.get("OAUTH_SLACK_CLIENT_ID")
    CLIENT_SECRET: str | None = os.environ.get("OAUTH_SLACK_CLIENT_SECRET")
    TOKEN_URL = "https://slack.com/api/oauth.v2.access"

    SLACK_OAUTH_SCOPE = "channels:history,channels:join,channels:read,groups:history,groups:read,im:history,im:read,mpim:history,mpim:read,users:read,users:read.email,team:read"

    class OAuthSession(BaseModel):
        email: str
        redirect_on_success: str | None

    @classmethod
    def _redirect_uri(cls, web_domain: str, dev_mode: bool) -> str:
        base = f"{web_domain}/admin/connectors/slack/oauth/callback"
        return f"https://redirectmeto.com/{base}" if dev_mode else base

    @classmethod
    def generate_oauth_url(
        cls, state: str, web_domain: str, dev_mode: bool = False
    ) -> str:
        redirect_uri = cls._redirect_uri(web_domain, dev_mode)
        return (
            "https://slack.com/oauth/v2/authorize"
            f"?client_id={cls.CLIENT_ID}"
            f"&scope={cls.SLACK_OAUTH_SCOPE}"
            f"&redirect_uri={redirect_uri}"
            f"&state={state}"
        )

    @classmethod
    def session_dump_json(cls, email: str, redirect_on_success: str | None) -> str:
        return cls.OAuthSession(
            email=email, redirect_on_success=redirect_on_success
        ).model_dump_json()

    @classmethod
    def parse_session(cls, session_json: str) -> "SlackOAuth.OAuthSession":
        return cls.OAuthSession.model_validate_json(session_json)


__all__ = ["SlackOAuth"]
