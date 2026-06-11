"""Shared OAuth provider primitives for connector authorization flows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar
from urllib.parse import quote, urlencode

from pydantic import BaseModel


class OAuthSession(BaseModel):
    email: str
    redirect_on_success: str | None


class ConnectorOAuthProvider(ABC):
    """Base implementation for connector OAuth URL and session handling."""

    PROVIDER_ID: ClassVar[str]
    DISPLAY_NAME: ClassVar[str]
    CLIENT_ID_ENV_VAR: ClassVar[str]
    CLIENT_ID: ClassVar[str | None]
    CLIENT_SECRET: ClassVar[str | None]
    AUTHORIZATION_URL: ClassVar[str]
    CALLBACK_SLUG: ClassVar[str]

    @classmethod
    def redirect_uri(cls, web_domain: str, dev_mode: bool) -> str:
        base = f"{web_domain}/admin/connectors/{cls.CALLBACK_SLUG}/oauth/callback"
        return f"https://redirectmeto.com/{base}" if dev_mode else base

    @classmethod
    @abstractmethod
    def authorization_params(
        cls,
        state: str,
        redirect_uri: str,
    ) -> dict[str, str]:
        raise NotImplementedError

    @classmethod
    def generate_oauth_url(
        cls,
        state: str,
        web_domain: str,
        dev_mode: bool = False,
    ) -> str:
        params = cls.authorization_params(
            state,
            cls.redirect_uri(web_domain, dev_mode),
        )
        return f"{cls.AUTHORIZATION_URL}?{urlencode(params, quote_via=quote)}"

    @classmethod
    def session_dump_json(cls, email: str, redirect_on_success: str | None) -> str:
        return OAuthSession(
            email=email,
            redirect_on_success=redirect_on_success,
        ).model_dump_json()

    @classmethod
    def parse_session(cls, session_json: str) -> OAuthSession:
        return OAuthSession.model_validate_json(session_json)
