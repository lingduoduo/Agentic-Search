"""Tests for src/servers/tenants and src/servers/user_group."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.auth import generate_user_jwt_token
from src.configs import AppSettings
from src.configs import AuthSettings
from src.db import AgenticSearchStore
from src.db.models import UserRecord
from src.servers.tenants.access import generate_data_plane_token
from src.servers.tenants.access import verify_data_plane_token
from src.servers.tenants.models import AnonymousUserPath
from src.servers.tenants.models import RequestInviteRequest
from src.servers.tenants.models import TierUpdateRequest
from src.servers.web.app import SearchExperienceSettings
from src.servers.web.app import create_web_app

_ADMIN = "admin-user"
_USER = "regular-user"


def _bearer(user_id: str) -> dict:
    return {"Authorization": f"Bearer {generate_user_jwt_token(user_id=user_id)}"}


def _make_client(tmp_path):
    store = AgenticSearchStore(tmp_path / "db.sqlite3")
    store.upsert_user(UserRecord(id=_ADMIN, email="admin@test.local"))
    store.upsert_user(UserRecord(id=_USER, email="user@test.local"))
    settings = AppSettings(auth=AuthSettings(super_users=(_ADMIN,)))
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"),
        app_settings=settings,
        store=store,
    )
    return TestClient(app), store


# ---------------------------------------------------------------------------
# tenants/access.py
# ---------------------------------------------------------------------------


def test_generate_data_plane_token_roundtrip():
    token = generate_data_plane_token()
    assert verify_data_plane_token(token) is True


def test_verify_data_plane_token_rejects_bogus():
    assert verify_data_plane_token("not.a.token") is False
    assert verify_data_plane_token("a.b.c") is False


def test_verify_data_plane_token_rejects_other_user_jwt():
    other_token = generate_user_jwt_token(user_id="someone-else")
    assert verify_data_plane_token(other_token) is False


def test_data_plane_token_expires(monkeypatch):
    import time

    token = generate_data_plane_token()
    # Advance time past expiry
    original_time = time.time
    monkeypatch.setattr("time.time", lambda: original_time() + 400)
    assert verify_data_plane_token(token) is False


# ---------------------------------------------------------------------------
# tenants/models.py
# ---------------------------------------------------------------------------


def test_tenant_models_instantiate():
    from src.configs import Tier

    req = TierUpdateRequest(tenant_id="t1", customer_tier=Tier.BUSINESS)
    assert req.trial_end is None

    path = AnonymousUserPath(anonymous_user_path="/anon")
    assert path.anonymous_user_path == "/anon"

    invite = RequestInviteRequest(email="a@b.com")
    assert invite.email == "a@b.com"


# ---------------------------------------------------------------------------
# tenants/api.py — 501 stubs
# ---------------------------------------------------------------------------


def test_tenants_billing_endpoints_return_501(tmp_path):
    client, store = _make_client(tmp_path)
    for path in [
        "/tenants/billing-information",
        "/tenants/stripe-publishable-key",
    ]:
        assert client.get(path).status_code == 501
    for path in [
        "/tenants/create-subscription-session",
        "/tenants/create-customer-portal-session",
    ]:
        assert client.post(path).status_code == 501
    store.close()


# ---------------------------------------------------------------------------
# user_group — store-level methods
# ---------------------------------------------------------------------------


def test_list_groups_empty(tmp_path):
    store = AgenticSearchStore(tmp_path / "db.sqlite3")
    assert store.list_groups() == []
    store.close()


def test_delete_group_returns_false_when_not_found(tmp_path):
    store = AgenticSearchStore(tmp_path / "db.sqlite3")
    assert store.delete_group("nonexistent") is False
    store.close()


def test_remove_user_from_group_is_idempotent(tmp_path):
    from src.db.models import GroupRecord

    store = AgenticSearchStore(tmp_path / "db.sqlite3")
    store.upsert_user(UserRecord(id="u1"))
    store.upsert_group(GroupRecord(id="g1", name="G1", user_ids=["u1"]))
    store.remove_user_from_group("u1", "g1")
    store.remove_user_from_group("u1", "g1")  # no-op
    group = store.get_group("g1")
    assert group is not None and "u1" not in group.user_ids
    store.close()


# ---------------------------------------------------------------------------
# user_group — API endpoints
# ---------------------------------------------------------------------------


def test_create_and_list_groups(tmp_path):
    client, store = _make_client(tmp_path)
    resp = client.post(
        "/manage/admin/user-group",
        json={"name": "Engineering", "user_ids": []},
        headers=_bearer(_ADMIN),
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Engineering"
    group_id = data["id"]

    resp = client.get("/manage/admin/user-group", headers=_bearer(_ADMIN))
    assert resp.status_code == 200
    assert any(g["id"] == group_id for g in resp.json())
    store.close()


def test_get_group_by_id(tmp_path):
    client, store = _make_client(tmp_path)
    resp = client.post(
        "/manage/admin/user-group",
        json={"name": "DevOps"},
        headers=_bearer(_ADMIN),
    )
    group_id = resp.json()["id"]
    resp = client.get(f"/manage/admin/user-group/{group_id}", headers=_bearer(_ADMIN))
    assert resp.status_code == 200
    assert resp.json()["name"] == "DevOps"
    store.close()


def test_get_group_404_for_missing(tmp_path):
    client, store = _make_client(tmp_path)
    resp = client.get("/manage/admin/user-group/noexist", headers=_bearer(_ADMIN))
    assert resp.status_code == 404
    store.close()


def test_update_group_name_and_members(tmp_path):
    client, store = _make_client(tmp_path)
    store.upsert_user(UserRecord(id="u1", email="u1@test.local"))
    resp = client.post(
        "/manage/admin/user-group",
        json={"name": "OldName"},
        headers=_bearer(_ADMIN),
    )
    group_id = resp.json()["id"]

    resp = client.patch(
        f"/manage/admin/user-group/{group_id}",
        json={"name": "NewName", "user_ids": ["u1"]},
        headers=_bearer(_ADMIN),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "NewName"
    assert "u1" in data["user_ids"]
    store.close()


def test_rename_group(tmp_path):
    client, store = _make_client(tmp_path)
    resp = client.post(
        "/manage/admin/user-group",
        json={"name": "Original"},
        headers=_bearer(_ADMIN),
    )
    group_id = resp.json()["id"]
    resp = client.patch(
        f"/manage/admin/user-group/{group_id}/rename",
        json={"name": "Renamed"},
        headers=_bearer(_ADMIN),
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "Renamed"
    store.close()


def test_add_and_remove_users(tmp_path):
    client, store = _make_client(tmp_path)
    store.upsert_user(UserRecord(id="u1", email="u1@t.local"))
    store.upsert_user(UserRecord(id="u2", email="u2@t.local"))
    resp = client.post(
        "/manage/admin/user-group",
        json={"name": "Team"},
        headers=_bearer(_ADMIN),
    )
    group_id = resp.json()["id"]

    client.post(
        f"/manage/admin/user-group/{group_id}/add-users",
        json={"user_ids": ["u1", "u2"]},
        headers=_bearer(_ADMIN),
    )
    group = client.get(
        f"/manage/admin/user-group/{group_id}", headers=_bearer(_ADMIN)
    ).json()
    assert set(group["user_ids"]) == {"u1", "u2"}

    client.post(
        f"/manage/admin/user-group/{group_id}/remove-users",
        json={"user_ids": ["u1"]},
        headers=_bearer(_ADMIN),
    )
    group = client.get(
        f"/manage/admin/user-group/{group_id}", headers=_bearer(_ADMIN)
    ).json()
    assert group["user_ids"] == ["u2"]
    store.close()


def test_delete_group(tmp_path):
    client, store = _make_client(tmp_path)
    resp = client.post(
        "/manage/admin/user-group",
        json={"name": "ToDelete"},
        headers=_bearer(_ADMIN),
    )
    group_id = resp.json()["id"]
    resp = client.delete(
        f"/manage/admin/user-group/{group_id}", headers=_bearer(_ADMIN)
    )
    assert resp.status_code == 204
    assert (
        client.get(
            f"/manage/admin/user-group/{group_id}", headers=_bearer(_ADMIN)
        ).status_code
        == 404
    )
    store.close()


def test_delete_group_404_for_missing(tmp_path):
    client, store = _make_client(tmp_path)
    resp = client.delete("/manage/admin/user-group/nope", headers=_bearer(_ADMIN))
    assert resp.status_code == 404
    store.close()


def test_group_endpoints_require_admin(tmp_path):
    client, store = _make_client(tmp_path)
    assert (
        client.get("/manage/admin/user-group", headers=_bearer(_USER)).status_code
        == 403
    )
    assert (
        client.post(
            "/manage/admin/user-group", json={"name": "X"}, headers=_bearer(_USER)
        ).status_code
        == 403
    )
    store.close()


def test_list_my_groups_returns_user_groups(tmp_path):
    client, store = _make_client(tmp_path)
    client.post(
        "/manage/admin/user-group",
        json={"name": "MyGroup", "user_ids": [_USER]},
        headers=_bearer(_ADMIN),
    )
    resp = client.get("/manage/user-groups", headers=_bearer(_USER))
    assert resp.status_code == 200
    assert any(g["name"] == "MyGroup" for g in resp.json())
    store.close()


def test_create_group_conflict_on_duplicate_id(tmp_path):
    client, store = _make_client(tmp_path)
    client.post(
        "/manage/admin/user-group",
        json={"name": "First", "group_id": "fixed-id"},
        headers=_bearer(_ADMIN),
    )
    resp = client.post(
        "/manage/admin/user-group",
        json={"name": "Second", "group_id": "fixed-id"},
        headers=_bearer(_ADMIN),
    )
    assert resp.status_code == 409
    store.close()
