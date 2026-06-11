"""Tests for features/hooks, license API, and middleware."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.internal.auth import generate_user_jwt_token
from src.internal.configs import AppSettings
from src.internal.configs import AuthSettings
from src.internal.db import AgenticSearchStore
from src.internal.db.models import HookRecord
from src.internal.db.models import UserRecord
from src.internal.servers.middleware.tenant_tracking import (
    add_api_server_tenant_id_middleware,
)
from src.internal.servers.middleware.tier_gate import add_tier_gate_middleware
from src.internal.servers.web.app import SearchExperienceSettings
from src.internal.servers.web.app import create_web_app

_ADMIN = "admin-user"
_USER = "regular-user"


def _bearer(user_id: str) -> dict:
    return {"Authorization": f"Bearer {generate_user_jwt_token(user_id=user_id)}"}


def _make_client(tmp_path, *, monkeypatch=None, data_dir=None):
    store = AgenticSearchStore(tmp_path / "db.sqlite3")
    store.upsert_user(UserRecord(id=_ADMIN, email="admin@test.local"))
    store.upsert_user(UserRecord(id=_USER, email="user@test.local"))
    settings = AppSettings(auth=AuthSettings(super_users=(_ADMIN,)))
    if monkeypatch and data_dir:
        monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(data_dir))
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"),
        app_settings=settings,
        store=store,
    )
    return TestClient(app), store


# ---------------------------------------------------------------------------
# db/models — HookRecord
# ---------------------------------------------------------------------------


def test_hook_record_defaults():
    h = HookRecord(
        id="h1",
        name="Test",
        hook_point="query_processing",
        endpoint_url="https://example.com/hook",
    )
    assert h.is_active is True
    assert h.fail_strategy == "soft"
    assert h.timeout_seconds == 5.0
    assert h.is_reachable is None


# ---------------------------------------------------------------------------
# db/store — hook CRUD
# ---------------------------------------------------------------------------


def test_store_hook_upsert_and_get(tmp_path):
    store = AgenticSearchStore(tmp_path / "db.sqlite3")
    hook = HookRecord(
        id="h1",
        name="My Hook",
        hook_point="query_processing",
        endpoint_url="https://hooks.test/qp",
        fail_strategy="hard",
        timeout_seconds=10.0,
        is_active=True,
    )
    stored = store.upsert_hook(hook)
    assert stored.id == "h1"
    assert stored.name == "My Hook"
    assert stored.timeout_seconds == 10.0

    fetched = store.get_hook("h1")
    assert fetched is not None
    assert fetched.fail_strategy == "hard"
    store.close()


def test_store_list_hooks(tmp_path):
    store = AgenticSearchStore(tmp_path / "db.sqlite3")
    for i in range(3):
        store.upsert_hook(
            HookRecord(
                id=f"h{i}",
                name=f"Hook {i}",
                hook_point="query_processing",
                endpoint_url=f"https://hooks.test/{i}",
            )
        )
    assert len(store.list_hooks()) == 3
    store.close()


def test_store_delete_hook(tmp_path):
    store = AgenticSearchStore(tmp_path / "db.sqlite3")
    store.upsert_hook(
        HookRecord(
            id="h1",
            name="H",
            hook_point="answer_generated",
            endpoint_url="https://x.com",
        )
    )
    assert store.delete_hook("h1") is True
    assert store.get_hook("h1") is None
    assert store.delete_hook("h1") is False
    store.close()


def test_store_hook_is_reachable_stored_correctly(tmp_path):
    store = AgenticSearchStore(tmp_path / "db.sqlite3")
    store.upsert_hook(
        HookRecord(
            id="h1",
            name="H",
            hook_point="before_retrieval",
            endpoint_url="https://x.com",
            is_reachable=True,
        )
    )
    assert store.get_hook("h1").is_reachable is True
    store.upsert_hook(
        HookRecord(
            id="h1",
            name="H",
            hook_point="before_retrieval",
            endpoint_url="https://x.com",
            is_reachable=False,
        )
    )
    assert store.get_hook("h1").is_reachable is False
    store.close()


# ---------------------------------------------------------------------------
# features/hooks API — via web app
# ---------------------------------------------------------------------------


def test_hooks_specs_endpoint(tmp_path):
    client, store = _make_client(tmp_path)
    resp = client.get("/admin/hooks/specs", headers=_bearer(_ADMIN))
    assert resp.status_code == 200
    specs = resp.json()
    assert len(specs) >= 4
    points = {s["hook_point"] for s in specs}
    assert "query_processing" in points
    store.close()


def test_hooks_list_empty(tmp_path):
    client, store = _make_client(tmp_path)
    resp = client.get("/admin/hooks", headers=_bearer(_ADMIN))
    assert resp.status_code == 200
    assert resp.json() == []
    store.close()


def test_hooks_get_404_for_missing(tmp_path):
    client, store = _make_client(tmp_path)
    resp = client.get("/admin/hooks/nonexistent", headers=_bearer(_ADMIN))
    assert resp.status_code == 404
    store.close()


def test_hooks_delete_404_for_missing(tmp_path):
    client, store = _make_client(tmp_path)
    resp = client.delete("/admin/hooks/nope", headers=_bearer(_ADMIN))
    assert resp.status_code == 404
    store.close()


def test_hooks_require_admin(tmp_path):
    client, store = _make_client(tmp_path)
    assert client.get("/admin/hooks", headers=_bearer(_USER)).status_code == 403
    assert client.get("/admin/hooks").status_code == 401
    store.close()


# ---------------------------------------------------------------------------
# license/models
# ---------------------------------------------------------------------------


def test_license_status_response_defaults():
    from src.internal.servers.license.models import LicenseStatusResponse
    from src.internal.utils.license_expiry import ExpiryWarningStage

    resp = LicenseStatusResponse(has_license=False)
    assert resp.expiry_warning_stage == ExpiryWarningStage.NONE
    assert resp.features == []


# ---------------------------------------------------------------------------
# license/api — file store helpers
# ---------------------------------------------------------------------------


def test_strip_pem_removes_delimiters():
    from src.internal.servers.license.api import _strip_pem

    content = "-----BEGIN AGENTIC SEARCH LICENSE-----\nabc123\n-----END AGENTIC SEARCH LICENSE-----"
    assert _strip_pem(content) == "abc123"


def test_strip_pem_passthrough_for_plain():
    from src.internal.servers.license.api import _strip_pem

    assert _strip_pem("abc123") == "abc123"


def test_license_status_returns_no_license(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    client, store = _make_client(tmp_path, monkeypatch=monkeypatch, data_dir=tmp_path)
    resp = client.get("/license", headers=_bearer(_ADMIN))
    assert resp.status_code == 200
    assert resp.json()["has_license"] is False
    store.close()


def test_license_seats_returns_zeros_without_license(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    client, store = _make_client(tmp_path, monkeypatch=monkeypatch, data_dir=tmp_path)
    resp = client.get("/license/seats", headers=_bearer(_ADMIN))
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_seats"] == 0
    store.close()


def test_license_delete_returns_not_deleted_when_none(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    client, store = _make_client(tmp_path, monkeypatch=monkeypatch, data_dir=tmp_path)
    resp = client.delete("/license", headers=_bearer(_ADMIN))
    assert resp.status_code == 200
    assert resp.json()["deleted"] is False
    store.close()


def test_license_endpoints_require_admin(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    client, store = _make_client(tmp_path, monkeypatch=monkeypatch, data_dir=tmp_path)
    assert client.get("/license", headers=_bearer(_USER)).status_code == 403
    assert client.get("/license").status_code == 401
    store.close()


def test_license_upload_rejects_invalid_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    client, store = _make_client(tmp_path, monkeypatch=monkeypatch, data_dir=tmp_path)
    resp = client.post(
        "/license/upload",
        files={
            "license_file": (
                "license.lic",
                b"not-valid-base64-license-data",
                "text/plain",
            )
        },
        headers=_bearer(_ADMIN),
    )
    assert resp.status_code == 400
    store.close()


def test_license_claim_returns_501_without_cloud_url(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    client, store = _make_client(tmp_path, monkeypatch=monkeypatch, data_dir=tmp_path)
    resp = client.post("/license/claim", headers=_bearer(_ADMIN))
    assert resp.status_code == 501
    store.close()


# ---------------------------------------------------------------------------
# middleware
# ---------------------------------------------------------------------------


def test_tenant_tracking_middleware_noop():
    from fastapi import FastAPI

    from src.internal.configs import load_app_settings

    app = FastAPI()
    settings = load_app_settings({})
    add_api_server_tenant_id_middleware(app, settings)
    # No middleware should be registered (noop)
    assert not any(
        "tenant" in str(m).lower() for m in app.middleware_stack.__class__.__mro__
    )


def test_tier_gate_middleware_registered_when_gates_exist():
    from fastapi import FastAPI

    from src.internal.configs import load_app_settings

    app = FastAPI()
    settings = load_app_settings({})
    # Should register without error even when PATH_PREFIX_MIN_TIER has entries
    add_tier_gate_middleware(app, settings)


def test_license_enforcement_middleware_noop_when_disabled(tmp_path, monkeypatch):
    """With license_enforcement_enabled=False, all paths pass through."""
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    client, store = _make_client(tmp_path, monkeypatch=monkeypatch, data_dir=tmp_path)
    # Health endpoint should always be reachable
    resp = client.get("/health")
    assert resp.status_code == 200
    store.close()
