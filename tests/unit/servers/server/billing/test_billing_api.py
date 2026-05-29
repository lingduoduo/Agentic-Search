"""Tests for the billing API endpoints (src/servers/billing/api.py).

All tests use TestClient so the full FastAPI dependency injection stack runs.
Stripe calls are mocked at the service layer to avoid real network requests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import patch


from .conftest import _ADMIN
from .conftest import _USER
from .conftest import _bearer
from .conftest import make_test_client


# ---------------------------------------------------------------------------
# Circuit-breaker unit tests (pure state, no HTTP)
# ---------------------------------------------------------------------------


def test_circuit_starts_closed() -> None:
    from src.servers.billing.api import _is_billing_circuit_open

    assert _is_billing_circuit_open() is False


def test_open_and_close_circuit() -> None:
    from src.servers.billing.api import (
        _close_billing_circuit,
        _is_billing_circuit_open,
        _open_billing_circuit,
    )

    _open_billing_circuit()
    assert _is_billing_circuit_open() is True
    _close_billing_circuit()
    assert _is_billing_circuit_open() is False


# ---------------------------------------------------------------------------
# /admin/billing/billing-information
# ---------------------------------------------------------------------------


def test_billing_info_returns_not_subscribed_without_license(tmp_path):
    client, store = make_test_client(tmp_path)
    with patch("src.servers.billing.api.load_stored_license", return_value=None):
        resp = client.get("/admin/billing/billing-information", headers=_bearer(_ADMIN))
    assert resp.status_code == 200
    assert resp.json()["subscribed"] is False
    store.close()


def test_billing_info_requires_admin(tmp_path):
    client, store = make_test_client(tmp_path)
    assert (
        client.get(
            "/admin/billing/billing-information", headers=_bearer(_USER)
        ).status_code
        == 403
    )
    assert client.get("/admin/billing/billing-information").status_code == 401
    store.close()


def test_billing_info_503_when_circuit_open(tmp_path):
    from src.servers.billing.api import _close_billing_circuit

    client, store = make_test_client(tmp_path)
    with (
        patch(
            "src.servers.billing.api.load_stored_license", return_value="license_blob"
        ),
        patch("src.servers.billing.api._is_billing_circuit_open", return_value=True),
    ):
        resp = client.get("/admin/billing/billing-information", headers=_bearer(_ADMIN))
    assert resp.status_code == 503
    _close_billing_circuit()  # clean up module-level state
    store.close()


def _make_cloud_client(tmp_path):
    """Make a TestClient with cloud_data_plane_url configured."""
    from src.configs import AppSettings, AuthSettings
    from src.db import AgenticSearchStore
    from src.db.models import UserRecord
    from src.servers.web.app import SearchExperienceSettings, create_web_app
    from fastapi.testclient import TestClient as _TestClient

    store = AgenticSearchStore(tmp_path / "db.sqlite3")
    store.upsert_user(UserRecord(id=_ADMIN, email="admin@test.local"))
    settings = AppSettings(
        auth=AuthSettings(super_users=(_ADMIN,)),
        cloud_data_plane_url="https://cloud.example.com",
    )
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"),
        app_settings=settings,
        store=store,
    )
    return _TestClient(app), store


def test_billing_info_opens_circuit_on_502(tmp_path):
    from fastapi import HTTPException

    from src.servers.billing.api import _close_billing_circuit, _is_billing_circuit_open

    client, store = _make_cloud_client(tmp_path)
    _close_billing_circuit()

    with (
        patch(
            "src.servers.billing.api.load_stored_license", return_value="license_blob"
        ),
        patch(
            "src.servers.billing.api._get_billing", new_callable=AsyncMock
        ) as mock_svc,
    ):
        mock_svc.side_effect = HTTPException(status_code=502, detail="upstream down")
        resp = client.get("/admin/billing/billing-information", headers=_bearer(_ADMIN))

    assert resp.status_code == 502
    assert _is_billing_circuit_open() is True
    _close_billing_circuit()
    store.close()


def test_billing_info_does_not_open_circuit_on_400(tmp_path):
    from fastapi import HTTPException

    from src.servers.billing.api import _close_billing_circuit, _is_billing_circuit_open

    client, store = _make_cloud_client(tmp_path)
    _close_billing_circuit()

    with (
        patch(
            "src.servers.billing.api.load_stored_license", return_value="license_blob"
        ),
        patch(
            "src.servers.billing.api._get_billing", new_callable=AsyncMock
        ) as mock_svc,
    ):
        mock_svc.side_effect = HTTPException(status_code=400, detail="bad request")
        resp = client.get("/admin/billing/billing-information", headers=_bearer(_ADMIN))

    assert resp.status_code == 400
    assert _is_billing_circuit_open() is False
    store.close()


# ---------------------------------------------------------------------------
# /admin/billing/create-checkout-session
# ---------------------------------------------------------------------------


def test_create_checkout_session_returns_501_without_cloud_url(tmp_path):
    client, store = make_test_client(tmp_path)
    with patch("src.servers.billing.api.load_stored_license", return_value=None):
        resp = client.post(
            "/admin/billing/create-checkout-session",
            json={"billing_period": "monthly"},
            headers=_bearer(_ADMIN),
        )
    assert resp.status_code == 501
    store.close()


def test_create_checkout_session_requires_admin(tmp_path):
    client, store = make_test_client(tmp_path)
    assert (
        client.post(
            "/admin/billing/create-checkout-session", json={}, headers=_bearer(_USER)
        ).status_code
        == 403
    )
    store.close()


# ---------------------------------------------------------------------------
# /admin/billing/create-customer-portal-session
# ---------------------------------------------------------------------------


def test_create_portal_session_returns_501_without_cloud_url(tmp_path):
    # Without cloud_data_plane_url, _require_proxy_url() raises 501 first
    client, store = make_test_client(tmp_path)
    with patch("src.servers.billing.api.load_stored_license", return_value=None):
        resp = client.post(
            "/admin/billing/create-customer-portal-session",
            json={},
            headers=_bearer(_ADMIN),
        )
    assert resp.status_code == 501
    store.close()


def test_create_portal_session_returns_400_when_no_license_with_cloud_url(tmp_path):
    # With cloud_data_plane_url set but no license → 400
    from src.configs import AppSettings, AuthSettings
    from src.db import AgenticSearchStore
    from src.db.models import UserRecord
    from src.servers.web.app import SearchExperienceSettings, create_web_app
    from fastapi.testclient import TestClient

    store = AgenticSearchStore(tmp_path / "db.sqlite3")
    store.upsert_user(UserRecord(id=_ADMIN, email="admin@test.local"))
    settings = AppSettings(
        auth=AuthSettings(super_users=(_ADMIN,)),
        cloud_data_plane_url="https://cloud.example.com",
    )
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"),
        app_settings=settings,
        store=store,
    )
    client = TestClient(app)

    with patch("src.servers.billing.api.load_stored_license", return_value=None):
        resp = client.post(
            "/admin/billing/create-customer-portal-session",
            json={},
            headers=_bearer(_ADMIN),
        )
    assert resp.status_code == 400
    store.close()


# ---------------------------------------------------------------------------
# /admin/billing/seats/update
# ---------------------------------------------------------------------------


def test_update_seats_returns_400_without_license(tmp_path):
    client, store = make_test_client(tmp_path)
    with patch("src.servers.billing.api.load_stored_license", return_value=None):
        resp = client.post(
            "/admin/billing/seats/update",
            json={"new_seat_count": 10},
            headers=_bearer(_ADMIN),
        )
    assert resp.status_code == 400
    store.close()


def test_update_seats_requires_admin(tmp_path):
    client, store = make_test_client(tmp_path)
    resp = client.post(
        "/admin/billing/seats/update",
        json={"new_seat_count": 10},
        headers=_bearer(_USER),
    )
    assert resp.status_code == 403
    store.close()


# ---------------------------------------------------------------------------
# /admin/billing/stripe-publishable-key
# ---------------------------------------------------------------------------


def test_stripe_key_from_override(tmp_path):
    from fastapi.testclient import TestClient as _TestClient

    from src.configs import AppSettings, AuthSettings
    from src.servers.web.app import SearchExperienceSettings, create_web_app

    store_path = tmp_path / "db.sqlite3"
    settings = AppSettings(
        auth=AuthSettings(super_users=(_ADMIN,)),
        stripe_publishable_key_override="pk_test_abc123",
    )
    app = create_web_app(
        SearchExperienceSettings(db_path=store_path),
        app_settings=settings,
    )
    client = _TestClient(app)
    # Reset module-level cache before test
    import src.servers.billing.api as billing_mod

    billing_mod._stripe_key_cache = None

    resp = client.get("/admin/billing/stripe-publishable-key")
    assert resp.status_code == 200
    assert resp.json()["publishable_key"] == "pk_test_abc123"
    billing_mod._stripe_key_cache = None  # clean up


def test_stripe_key_returns_500_when_not_configured(tmp_path):
    client, store = make_test_client(tmp_path)
    import src.servers.billing.api as billing_mod

    billing_mod._stripe_key_cache = None

    resp = client.get("/admin/billing/stripe-publishable-key")
    assert resp.status_code == 500
    billing_mod._stripe_key_cache = None
    store.close()


# ---------------------------------------------------------------------------
# /admin/billing/reset-connection
# ---------------------------------------------------------------------------


def test_reset_connection_closes_circuit(tmp_path):
    from src.servers.billing.api import _open_billing_circuit, _is_billing_circuit_open

    _open_billing_circuit()
    assert _is_billing_circuit_open() is True

    client, store = make_test_client(tmp_path)
    resp = client.post("/admin/billing/reset-connection", headers=_bearer(_ADMIN))
    assert resp.status_code == 200
    assert "re-enabled" in resp.json()["message"].lower()
    assert _is_billing_circuit_open() is False
    store.close()


def test_reset_connection_requires_admin(tmp_path):
    client, store = make_test_client(tmp_path)
    assert (
        client.post(
            "/admin/billing/reset-connection", headers=_bearer(_USER)
        ).status_code
        == 403
    )
    store.close()
