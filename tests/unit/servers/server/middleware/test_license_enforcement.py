"""Tests for the license enforcement middleware (src/servers/middleware/license_enforcement.py).

Our middleware is simpler than the sampled version: it reads the license from a
file (no Redis, no DB) and uses AppSettings.license_enforcement_enabled.
"""

from __future__ import annotations


import pytest

from src.backend.servers.middleware.license_enforcement import _load_payload


class TestIsPathAllowedForTier:
    """The path allowlist is driven by LICENSE_ENFORCEMENT_ALLOWED_PREFIXES."""

    @pytest.mark.parametrize(
        "path",
        ["/auth/callback", "/health", "/assets/app.js", "/admin/billing/checkout"],
    )
    def test_allowed_paths_pass(self, path: str) -> None:
        from src.backend.configs import is_license_enforcement_exempt

        assert is_license_enforcement_exempt(path) is True

    @pytest.mark.parametrize("path", ["/api/chat", "/search", "/admin/connectors"])
    def test_gated_paths_are_not_exempt(self, path: str) -> None:
        from src.backend.configs import is_license_enforcement_exempt

        assert is_license_enforcement_exempt(path) is False


class TestLoadPayload:
    """_load_payload reads from the license file; no file → None."""

    def test_returns_none_without_license_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
        assert _load_payload() is None

    def test_returns_none_for_invalid_license(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
        (tmp_path / "license.dat").write_text("not-valid-base64!!!")
        assert _load_payload() is None


class TestLicenseEnforcementMiddlewareIntegration:
    """End-to-end tests via TestClient with enforcement enabled/disabled."""

    def _make_client(self, tmp_path, *, enforcement: bool = False):
        from fastapi.testclient import TestClient

        from src.backend.auth import generate_user_jwt_token
        from src.backend.configs import AppSettings, AuthSettings
        from src.backend.db import AgenticSearchStore
        from src.backend.db.models import UserRecord
        from src.backend.servers.web.app import SearchExperienceSettings, create_web_app

        store = AgenticSearchStore(tmp_path / "db.sqlite3")
        store.upsert_user(UserRecord(id="admin", email="admin@test.local"))
        settings = AppSettings(
            auth=AuthSettings(super_users=("admin",)),
            license_enforcement_enabled=enforcement,
        )
        app = create_web_app(
            SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"),
            app_settings=settings,
            store=store,
        )
        token = generate_user_jwt_token(user_id="admin")
        return TestClient(app), store, token

    def test_health_always_reachable(self, tmp_path) -> None:
        client, store, _ = self._make_client(tmp_path, enforcement=True)
        assert client.get("/health").status_code == 200
        store.close()

    def test_enforcement_disabled_allows_all_paths(self, tmp_path) -> None:
        client, store, token = self._make_client(tmp_path, enforcement=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        store.close()

    def test_enforcement_enabled_without_license_returns_402_for_gated_paths(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
        # No license.dat file → middleware should gate
        client, store, token = self._make_client(tmp_path, enforcement=True)
        # /analytics is a gated path
        resp = client.get(
            "/analytics/query",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 402
        store.close()

    def test_enforcement_enabled_allows_exempt_paths(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
        client, store, _ = self._make_client(tmp_path, enforcement=True)
        assert client.get("/health").status_code == 200
        store.close()
