"""SCIM token administration requires admin, not merely a request.

`POST /scim/v2/tokens` had no authentication and returned a working bearer
token. That is not a disclosure bug like the session one -- it is a full
authentication bypass of the rest of this router, because the minted token then
satisfies the guard on `/scim/v2/Users` and `/scim/v2/Groups`:

    GET  /scim/v2/Users        (no creds)     -> 401
    POST /scim/v2/tokens       (no creds)     -> 201 + raw_token
    GET  /scim/v2/Users        (minted token) -> 200
    POST /scim/v2/Users        (minted token) -> 201  (provisioned a user)
    DELETE /scim/v2/Users/{id} (minted token) -> 204  (deprovisioned them)

`GET`/`DELETE` on tokens were unguarded too, so an anonymous caller could also
enumerate integrations and revoke their credentials.

The auth was never missing from the router -- the directory endpoints have
always used it. It was missing from three handlers, which is the same shape as
the session-ownership gap: per-handler auth with no mechanism that notices a
handler without it.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


@pytest.fixture()
def client(monkeypatch):
    # The repo's .env sets AGENTIC_SEARCH_DEV_ADMIN=1, which makes every admin
    # guard return a synthetic admin. Leaving it on would make these tests pass
    # against a completely unguarded router.
    monkeypatch.setitem(os.environ, "AGENTIC_SEARCH_DEV_ADMIN", "false")
    directory = Path(tempfile.mkdtemp())
    app = create_web_app(SearchExperienceSettings(db_path=directory / "scim.sqlite3"))
    with TestClient(app) as test_client:
        yield test_client


def test_minting_a_token_requires_admin(client):
    """The escalation itself: an anonymous mint is a bypass of the whole router."""
    response = client.post("/scim/v2/tokens", json={"name": "attacker-minted"})

    assert response.status_code in (401, 403), (
        f"anonymous caller minted a SCIM token: {response.status_code} "
        f"{response.text[:200]}"
    )
    assert "raw_token" not in response.text


def test_listing_tokens_requires_admin(client):
    """Metadata only -- `token_display` is masked -- but it enumerates integrations."""
    assert client.get("/scim/v2/tokens").status_code in (401, 403)


def test_revoking_a_token_requires_admin(client):
    """Unguarded revocation is a denial of service against real integrations."""
    assert client.delete("/scim/v2/tokens/any").status_code in (401, 403)


def test_directory_endpoints_are_unchanged(client):
    """The guard that always worked must keep working."""
    assert client.get("/scim/v2/Users").status_code == 401
    assert client.get("/scim/v2/Groups").status_code == 401


def test_scim_discovery_stays_public(client):
    """SCIM 2.0 requires these unauthenticated; narrowing them would break clients.

    Pinned so a later "guard everything under /scim" sweep does not quietly
    break provisioning integrations that read these before authenticating.
    """
    for path in (
        "/scim/v2/ServiceProviderConfig",
        "/scim/v2/Schemas",
        "/scim/v2/ResourceTypes",
    ):
        assert client.get(path).status_code == 200, path
