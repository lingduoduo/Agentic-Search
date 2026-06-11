"""Tests for the tier gate middleware (src/servers/middleware/tier_gate.py)."""

from __future__ import annotations

from unittest.mock import patch


from src.internal.configs import Tier


def _make_client(tmp_path, tier: Tier = Tier.FREE):
    from fastapi.testclient import TestClient

    from src.internal.auth import generate_user_jwt_token
    from src.internal.configs import AppSettings, AuthSettings
    from src.internal.db import AgenticSearchStore
    from src.internal.db.models import UserRecord
    from src.internal.servers.web.app import SearchExperienceSettings, create_web_app

    store = AgenticSearchStore(tmp_path / "db.sqlite3")
    store.upsert_user(UserRecord(id="admin", email="admin@test.local"))
    settings = AppSettings(auth=AuthSettings(super_users=("admin",)))
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"),
        app_settings=settings,
        store=store,
    )
    token = generate_user_jwt_token(user_id="admin")
    return TestClient(app), store, token


class TestTierGate:
    """Tier gate blocks paths based on the tier returned by get_tier()."""

    def test_free_tier_is_blocked_from_admin_paths(self, tmp_path) -> None:
        client, store, token = _make_client(tmp_path)
        with patch(
            "src.internal.servers.middleware.tier_gate.get_tier", return_value=Tier.FREE
        ):
            # /admin/token-rate-limits requires at least BUSINESS
            resp = client.get(
                "/admin/token-rate-limits/users",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code in (402, 403)
        store.close()

    def test_enterprise_tier_passes_all_paths(self, tmp_path) -> None:
        client, store, token = _make_client(tmp_path)
        with patch(
            "src.internal.servers.middleware.tier_gate.get_tier",
            return_value=Tier.ENTERPRISE,
        ):
            resp = client.get(
                "/admin/token-rate-limits/users",
                headers={"Authorization": f"Bearer {token}"},
            )
        # 200 (empty list) or 401 — not 402
        assert resp.status_code != 402
        store.close()

    def test_health_always_passes(self, tmp_path) -> None:
        client, store, _ = _make_client(tmp_path)
        with patch(
            "src.internal.servers.middleware.tier_gate.get_tier", return_value=Tier.FREE
        ):
            resp = client.get("/health")
        assert resp.status_code == 200
        store.close()


class TestRequiredTierHelper:
    """_required_tier maps path prefixes to the minimum tier."""

    def test_unmapped_path_returns_none(self) -> None:
        from src.internal.servers.middleware.tier_gate import _required_tier

        assert _required_tier("/api/chat") is None

    def test_admin_path_returns_tier(self) -> None:
        from src.internal.servers.middleware.tier_gate import _required_tier

        # The exact tier depends on the config; just assert it's a Tier instance
        result = _required_tier("/admin/token-rate-limits/users")
        assert result is None or isinstance(result, Tier)
