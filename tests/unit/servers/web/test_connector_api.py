"""HTTP-level tests for /admin/connectors/* endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.internal.auth import generate_user_jwt_token
from src.internal.configs import AppSettings, AuthSettings
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app

_ADMIN = "admin"


def _admin_headers() -> dict[str, str]:
    token = generate_user_jwt_token(user_id=_ADMIN)
    return {"Authorization": f"Bearer {token}"}


def _settings() -> AppSettings:
    return AppSettings(auth=AuthSettings(super_users=(_ADMIN,)))


def _make_app(tmp_path):
    return create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"),
        app_settings=_settings(),
    )


# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------


def test_list_connectors_requires_auth(tmp_path):
    client = TestClient(_make_app(tmp_path))
    assert client.get("/admin/connectors").status_code == 401


def test_create_connector_requires_auth(tmp_path):
    client = TestClient(_make_app(tmp_path))
    assert (
        client.post(
            "/admin/connectors", json={"name": "X", "source": "web"}
        ).status_code
        == 401
    )


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


def test_list_connectors_empty(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.get("/admin/connectors", headers=_admin_headers())
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def test_create_connector_returns_201(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.post(
        "/admin/connectors",
        json={
            "name": "My Web",
            "source": "web",
            "config": {"urls": ["https://example.test"]},
        },
        headers=_admin_headers(),
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"].startswith("connector_")
    assert data["name"] == "My Web"
    assert data["source"] == "web"
    assert data["enabled"] is True


def test_create_connector_duplicate_name_returns_409(tmp_path):
    client = TestClient(_make_app(tmp_path))
    payload = {"name": "Dup", "source": "file"}
    client.post("/admin/connectors", json=payload, headers=_admin_headers())
    resp = client.post("/admin/connectors", json=payload, headers=_admin_headers())
    assert resp.status_code == 409


def test_list_connectors_returns_created(tmp_path):
    client = TestClient(_make_app(tmp_path))
    client.post(
        "/admin/connectors",
        json={"name": "Slack", "source": "slack"},
        headers=_admin_headers(),
    )
    resp = client.get("/admin/connectors", headers=_admin_headers())
    assert resp.status_code == 200
    names = [c["name"] for c in resp.json()]
    assert "Slack" in names


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------


def test_get_connector_returns_detail(tmp_path):
    client = TestClient(_make_app(tmp_path))
    created = client.post(
        "/admin/connectors",
        json={
            "name": "RSS",
            "source": "rss",
            "config": {"feed_url": "https://feed.test"},
        },
        headers=_admin_headers(),
    ).json()
    resp = client.get(f"/admin/connectors/{created['id']}", headers=_admin_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == created["id"]
    assert data["document_count"] == 0
    assert isinstance(data["attempts"], list)


def test_get_unknown_connector_returns_404(tmp_path):
    client = TestClient(_make_app(tmp_path))
    assert (
        client.get(
            "/admin/connectors/nonexistent", headers=_admin_headers()
        ).status_code
        == 404
    )


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


def test_update_connector_name_and_enabled(tmp_path):
    client = TestClient(_make_app(tmp_path))
    created = client.post(
        "/admin/connectors",
        json={"name": "Old Name", "source": "file"},
        headers=_admin_headers(),
    ).json()
    resp = client.patch(
        f"/admin/connectors/{created['id']}",
        json={"name": "New Name", "enabled": False},
        headers=_admin_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "New Name"
    assert data["enabled"] is False


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_connector_returns_204(tmp_path):
    client = TestClient(_make_app(tmp_path))
    created = client.post(
        "/admin/connectors",
        json={"name": "ToDelete", "source": "web"},
        headers=_admin_headers(),
    ).json()
    assert (
        client.delete(
            f"/admin/connectors/{created['id']}", headers=_admin_headers()
        ).status_code
        == 204
    )
    assert (
        client.get(
            f"/admin/connectors/{created['id']}", headers=_admin_headers()
        ).status_code
        == 404
    )


def test_delete_unknown_connector_returns_404(tmp_path):
    client = TestClient(_make_app(tmp_path))
    assert (
        client.delete("/admin/connectors/ghost", headers=_admin_headers()).status_code
        == 404
    )


# ---------------------------------------------------------------------------
# Run (enqueue attempt)
# ---------------------------------------------------------------------------


def test_run_connector_enqueues_attempt(tmp_path):
    client = TestClient(_make_app(tmp_path))
    created = client.post(
        "/admin/connectors",
        json={"name": "SyncMe", "source": "web"},
        headers=_admin_headers(),
    ).json()
    resp = client.post(
        f"/admin/connectors/{created['id']}/run", headers=_admin_headers()
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["connector_id"] == created["id"]
    assert data["attempt_id"]


def test_run_unknown_connector_returns_404(tmp_path):
    client = TestClient(_make_app(tmp_path))
    assert (
        client.post("/admin/connectors/ghost/run", headers=_admin_headers()).status_code
        == 404
    )


# ---------------------------------------------------------------------------
# Ingest (no admin auth required)
# ---------------------------------------------------------------------------


def test_ingest_no_connector_id_creates_shared_connector(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.post(
        "/admin/connectors/ingest",
        json={
            "documents": [
                {
                    "title": "Doc A",
                    "contents": "Content of doc A.",
                    "url": "https://a.test",
                }
            ]
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ingested"] == 1
    assert data["connector_id"]


def test_ingest_second_call_reuses_shared_connector(tmp_path):
    client = TestClient(_make_app(tmp_path))
    r1 = client.post(
        "/admin/connectors/ingest",
        json={"documents": [{"title": "A", "contents": "a"}]},
    ).json()
    r2 = client.post(
        "/admin/connectors/ingest",
        json={"documents": [{"title": "B", "contents": "b"}]},
    ).json()
    assert r1["connector_id"] == r2["connector_id"]


def test_ingest_with_explicit_connector_id(tmp_path):
    client = TestClient(_make_app(tmp_path))
    created = client.post(
        "/admin/connectors",
        json={"name": "PushTarget", "source": "ingestion_api"},
        headers=_admin_headers(),
    ).json()
    resp = client.post(
        "/admin/connectors/ingest",
        json={
            "connector_id": created["id"],
            "documents": [
                {"title": "Doc B", "contents": "Content B."},
                {"title": "Doc C", "contents": "Content C."},
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["ingested"] == 2
    assert resp.json()["connector_id"] == created["id"]


def test_ingest_unknown_connector_returns_404(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.post(
        "/admin/connectors/ingest",
        json={"connector_id": "ghost", "documents": [{"title": "X", "contents": "x"}]},
    )
    assert resp.status_code == 404
