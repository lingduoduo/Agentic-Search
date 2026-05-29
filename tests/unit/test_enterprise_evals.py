"""Tests for src/servers/enterprise_settings and src/servers/evals."""

from __future__ import annotations


import pytest
from fastapi.testclient import TestClient

from src.auth import generate_user_jwt_token
from src.configs import AppSettings
from src.configs import AuthSettings
from src.db import AgenticSearchStore
from src.db.models import UserRecord
from src.servers.enterprise_settings.models import AnalyticsScriptUpload
from src.servers.enterprise_settings.models import EnterpriseSettings
from src.servers.enterprise_settings.models import NavigationItem
from src.servers.enterprise_settings.store import load_runtime_settings
from src.servers.enterprise_settings.store import load_settings
from src.servers.enterprise_settings.store import store_settings
from src.servers.web.app import SearchExperienceSettings
from src.servers.web.app import create_web_app

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
# enterprise_settings/models.py
# ---------------------------------------------------------------------------


def test_enterprise_settings_defaults():
    s = EnterpriseSettings()
    assert s.application_name is None
    assert s.use_custom_logo is False
    assert s.custom_nav_items == []


def test_navigation_item_requires_exactly_one_of_icon_or_svg():
    # Both missing → error
    with pytest.raises(Exception):
        NavigationItem.model_validate({"link": "/", "title": "Home"})
    # Both present → error
    with pytest.raises(Exception):
        NavigationItem.model_validate(
            {"link": "/", "title": "Home", "icon": "fa-home", "svg_logo": "<svg/>"}
        )
    # icon only → ok
    item = NavigationItem(link="/", title="Home", icon="fa-home")
    assert item.icon == "fa-home"
    # svg only → ok
    item2 = NavigationItem(link="/a", title="A", svg_logo="<svg/>")
    assert item2.svg_logo == "<svg/>"


def test_analytics_script_upload_model():
    upload = AnalyticsScriptUpload(script="console.log(1)", secret_key="s3cr3t")
    assert upload.script == "console.log(1)"


# ---------------------------------------------------------------------------
# enterprise_settings/store.py
# ---------------------------------------------------------------------------


def test_load_settings_returns_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    settings = load_settings()
    assert isinstance(settings, EnterpriseSettings)
    assert settings.application_name is None


def test_store_and_load_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    s = EnterpriseSettings(application_name="MyApp", use_custom_logo=True)
    store_settings(s)
    loaded = load_settings()
    assert loaded.application_name == "MyApp"
    assert loaded.use_custom_logo is True


def test_load_runtime_settings_applies_default_name(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    runtime = load_runtime_settings()
    assert runtime.application_name == "Agentic Search"


def test_load_runtime_settings_keeps_custom_name(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    store_settings(EnterpriseSettings(application_name="Custom"))
    runtime = load_runtime_settings()
    assert runtime.application_name == "Custom"


# ---------------------------------------------------------------------------
# enterprise_settings/api.py — via web app
# ---------------------------------------------------------------------------


def test_fetch_settings_is_public(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    client, store = _make_client(tmp_path, monkeypatch=monkeypatch, data_dir=tmp_path)
    response = client.get("/enterprise-settings")
    assert response.status_code == 200
    data = response.json()
    assert "application_name" in data
    store.close()


def test_put_settings_requires_admin(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    client, store = _make_client(tmp_path, monkeypatch=monkeypatch, data_dir=tmp_path)
    resp = client.put(
        "/admin/enterprise-settings",
        json={"application_name": "Test"},
        headers=_bearer(_USER),
    )
    assert resp.status_code == 403
    store.close()


def test_put_settings_stores_and_retrieves(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    client, store = _make_client(tmp_path, monkeypatch=monkeypatch, data_dir=tmp_path)
    client.put(
        "/admin/enterprise-settings",
        json={"application_name": "Branded App", "use_custom_logo": True},
        headers=_bearer(_ADMIN),
    )
    response = client.get("/enterprise-settings")
    data = response.json()
    assert data["application_name"] == "Branded App"
    assert data["use_custom_logo"] is True
    store.close()


def test_analytics_script_rejects_wrong_key(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CUSTOM_ANALYTICS_SECRET_KEY", "correct-key")
    client, store = _make_client(tmp_path, monkeypatch=monkeypatch, data_dir=tmp_path)
    resp = client.put(
        "/admin/enterprise-settings/custom-analytics-script",
        json={"script": "console.log(1)", "secret_key": "wrong-key"},
        headers=_bearer(_ADMIN),
    )
    assert resp.status_code == 400
    store.close()


def test_analytics_script_accepts_correct_key(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CUSTOM_ANALYTICS_SECRET_KEY", "my-secret")
    client, store = _make_client(tmp_path, monkeypatch=monkeypatch, data_dir=tmp_path)
    resp = client.put(
        "/admin/enterprise-settings/custom-analytics-script",
        json={"script": "console.log('hello')", "secret_key": "my-secret"},
        headers=_bearer(_ADMIN),
    )
    assert resp.status_code == 200
    get_resp = client.get("/enterprise-settings/custom-analytics-script")
    assert "hello" in get_resp.text
    store.close()


def test_logo_404_when_not_uploaded(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    client, store = _make_client(tmp_path, monkeypatch=monkeypatch, data_dir=tmp_path)
    assert client.get("/enterprise-settings/logo").status_code == 404
    store.close()


def test_scim_and_refresh_endpoints_return_501(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    client, store = _make_client(tmp_path, monkeypatch=monkeypatch, data_dir=tmp_path)
    assert (
        client.get(
            "/admin/enterprise-settings/scim/token", headers=_bearer(_ADMIN)
        ).status_code
        == 501
    )
    assert (
        client.post(
            "/admin/enterprise-settings/scim/token", headers=_bearer(_ADMIN)
        ).status_code
        == 501
    )
    assert client.post("/enterprise-settings/refresh-token").status_code == 501
    store.close()


# ---------------------------------------------------------------------------
# evals/api.py
# ---------------------------------------------------------------------------


def test_eval_run_requires_admin(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    client, store = _make_client(tmp_path)
    resp = client.post(
        "/evals/eval_run",
        json={"query": "test"},
        headers=_bearer(_USER),
    )
    assert resp.status_code == 403
    store.close()


def test_eval_run_returns_result_on_search_failure(tmp_path, monkeypatch):
    """When the retrieval server is unreachable, eval returns success=False gracefully."""
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    client, store = _make_client(tmp_path)
    resp = client.post(
        "/evals/eval_run",
        json={"query": "what is ML?", "num_hits": 3},
        headers=_bearer(_ADMIN),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "success" in data
    assert "query" in data
    assert data["query"] == "what is ML?"
    store.close()


def test_eval_run_with_fake_search(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))

    async def fake_run_expanded_search(query, **_):
        from src.retrieval.context import SearchResult
        from src.search.process_search_query import SearchQueryResult

        return SearchQueryResult(
            original_query=query,
            executed_queries=[query],
            results=[
                SearchResult(contents="ML is machine learning", score=0.9, title="ML")
            ],
        )

    monkeypatch.setattr(
        "src.servers.evals.api.run_expanded_search", fake_run_expanded_search
    )
    client, store = _make_client(tmp_path)
    resp = client.post(
        "/evals/eval_run",
        json={"query": "what is ML?"},
        headers=_bearer(_ADMIN),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["num_results"] == 1
    assert data["results"][0]["title"] == "ML"
    store.close()


def test_eval_run_ack_returns_immediately(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
    client, store = _make_client(tmp_path)
    resp = client.post(
        "/evals/eval_run_ack",
        json={"query": "test query"},
        headers=_bearer(_ADMIN),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "queued" in data.get("message", "").lower()
    store.close()
