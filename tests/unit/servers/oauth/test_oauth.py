from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.auth import generate_user_jwt_token
from src.internal.configs import AppSettings, AuthSettings
from src.internal.servers.oauth import OAUTH_PROVIDERS, OAuthSession
from src.internal.servers.oauth.api import create_oauth_router

_ADMIN_ID = "admin-user"


def _admin_headers() -> dict[str, str]:
    token = generate_user_jwt_token(user_id=_ADMIN_ID, email="admin@example.test")
    return {"Authorization": f"Bearer {token}"}


def _client() -> TestClient:
    settings = AppSettings(auth=AuthSettings(super_users=(_ADMIN_ID,)))
    app = FastAPI()
    app.include_router(create_oauth_router(settings))
    return TestClient(app)


@pytest.mark.parametrize("provider_id", sorted(OAUTH_PROVIDERS))
def test_oauth_providers_share_url_and_session_behavior(
    provider_id: str,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = OAUTH_PROVIDERS[provider_id]
    monkeypatch.setattr(provider, "CLIENT_ID", "client-id")

    parsed_url = urlparse(
        provider.generate_oauth_url("state-value", "https://app.example.test")
    )
    params = parse_qs(parsed_url.query)

    assert params["client_id"] == ["client-id"]
    assert params["state"] == ["state-value"]
    assert params["redirect_uri"] == [
        f"https://app.example.test/admin/connectors/{provider.CALLBACK_SLUG}/oauth/callback"
    ]
    assert provider.parse_session(
        provider.session_dump_json("admin@example.test", "/connectors")
    ) == OAuthSession(
        email="admin@example.test",
        redirect_on_success="/connectors",
    )


def test_prepare_authorization_request_uses_provider_registry(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = OAUTH_PROVIDERS["slack"]
    monkeypatch.setattr(provider, "CLIENT_ID", "client-id")

    response = _client().post(
        "/oauth/prepare-authorization-request",
        params={"connector": "SLACK", "redirect_on_success": "/connectors"},
        headers=_admin_headers(),
    )

    assert response.status_code == 200
    assert parse_qs(urlparse(response.json()["url"]).query)["client_id"] == [
        "client-id"
    ]


def test_prepare_authorization_request_rejects_unknown_provider():
    response = _client().post(
        "/oauth/prepare-authorization-request",
        params={"connector": "unknown"},
        headers=_admin_headers(),
    )

    assert response.status_code == 404
    assert f"Supported: {sorted(OAUTH_PROVIDERS)}" in response.json()["detail"]


def test_prepare_authorization_request_reports_missing_client_id(
    monkeypatch: pytest.MonkeyPatch,
):
    provider = OAUTH_PROVIDERS["google_drive"]
    monkeypatch.setattr(provider, "CLIENT_ID", None)

    response = _client().post(
        "/oauth/prepare-authorization-request",
        params={"connector": "google_drive"},
        headers=_admin_headers(),
    )

    assert response.status_code == 500
    assert provider.CLIENT_ID_ENV_VAR in response.json()["detail"]
