from __future__ import annotations

from fastapi.testclient import TestClient

from src.internal.auth import generate_user_jwt_token
from src.internal.configs import AppSettings
from src.internal.configs import AuthSettings
from src.internal.db import AgenticSearchStore
from src.internal.db import ConnectorConfig
from src.internal.db import GroupRecord
from src.internal.db import StoredDocument
from src.internal.db import UserRecord
from src.internal.db.models import HookRecord
from src.internal.observability import build_admin_surface_summary
from src.internal.servers.web.app import SearchExperienceSettings
from src.internal.servers.web.app import create_web_app

_ADMIN = "admin"


def _admin_headers() -> dict[str, str]:
    token = generate_user_jwt_token(user_id=_ADMIN)
    return {"Authorization": f"Bearer {token}"}


def _settings() -> AppSettings:
    return AppSettings(auth=AuthSettings(super_users=(_ADMIN,)))


def _seed_admin_surface(store: AgenticSearchStore) -> None:
    store.upsert_connector(
        ConnectorConfig(id="slack", name="Slack", source="slack", enabled=True)
    )
    store.upsert_connector(
        ConnectorConfig(id="drive", name="Drive", source="google_drive", enabled=False)
    )
    store.upsert_document(
        StoredDocument(
            id="doc-1",
            title="Runbook",
            contents="Deploy runbook",
            connector_id="slack",
        )
    )
    store.create_index_attempt(
        connector_id="slack",
        status="success",
        total_documents=1,
        total_chunks=2,
    )
    store.upsert_user(UserRecord(id="alice", email="alice@example.test"))
    store.upsert_group(GroupRecord(id="eng", name="Engineering", user_ids=["alice"]))
    store.create_scim_user_mapping("alice", external_id="okta-alice")
    store.create_scim_group_mapping("eng", external_id="okta-eng")
    store.upsert_hook(
        HookRecord(
            id="hook-1",
            name="Audit hook",
            hook_point="query_processing",
            endpoint_url="https://hooks.example.test/query",
        )
    )


def test_admin_surface_summary_rolls_up_operational_state(tmp_path):
    store = AgenticSearchStore(tmp_path / "state.sqlite3")
    try:
        _seed_admin_surface(store)

        summary = build_admin_surface_summary(store, _settings())

        assert summary.health_score == 100
        assert summary.metrics[0].label == "Connectors"
        assert summary.metrics[0].value == "1"
        assert summary.metrics[1].value == "1"
        assert summary.metrics[2].detail == "2 SCIM mapped"
        assert {section.key for section in summary.sections} >= {
            "connectors",
            "indexing",
            "access",
            "auth",
            "models",
            "tools",
            "analytics",
            "enterprise",
        }
    finally:
        store.close()


def test_admin_observability_endpoint_requires_admin(tmp_path):
    store = AgenticSearchStore(tmp_path / "state.sqlite3")
    try:
        _seed_admin_surface(store)
        app = create_web_app(
            SearchExperienceSettings(db_path=tmp_path / "unused.sqlite3"),
            app_settings=_settings(),
            store=store,
        )
        client = TestClient(app)

        unauthorized = client.get("/admin/observability/summary")
        authorized = client.get(
            "/admin/observability/summary", headers=_admin_headers()
        )

        assert unauthorized.status_code == 401
        assert authorized.status_code == 200
        data = authorized.json()
        assert data["healthScore"] == 100
        assert data["sections"][0]["key"] == "connectors"
    finally:
        store.close()
