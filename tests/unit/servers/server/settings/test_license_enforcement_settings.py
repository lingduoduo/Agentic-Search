"""Tests for the settings API (src/servers/settings/api.py).

Our settings API is simpler than the sampled version: no Redis, no DB — it
reads the license file directly via load_stored_license() and returns a
Settings model with tier / ee_features_enabled / application_status.
"""

from __future__ import annotations


from src.configs import Tier
from src.servers.settings.api import Settings
from src.servers.settings.api import _load_license_status
from src.utils.license import ApplicationStatus


def _enforcement_enabled_settings():
    from src.configs import AppSettings

    return AppSettings(license_enforcement_enabled=True)


def _enforcement_disabled_settings():
    from src.configs import AppSettings

    return AppSettings(license_enforcement_enabled=False)


class TestLoadLicenseStatus:
    """_load_license_status(app_settings) → (status, ee_on, tier_value)"""

    def test_enforcement_disabled_returns_enterprise_tier(self) -> None:
        settings = _enforcement_disabled_settings()
        status, ee_on, tier_val = _load_license_status(settings)
        assert status is None
        assert ee_on is True  # enforcement off → full access
        assert tier_val == Tier.ENTERPRISE.value

    def test_enforcement_enabled_no_license_returns_gated(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
        settings = _enforcement_enabled_settings()
        status, ee_on, tier_val = _load_license_status(settings)
        assert status == ApplicationStatus.GATED_ACCESS
        assert ee_on is False
        assert tier_val == Tier.FREE.value

    def test_enforcement_enabled_invalid_license_returns_gated(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
        (tmp_path / "license.dat").write_text("not-valid-base64!!!")
        settings = _enforcement_enabled_settings()
        status, ee_on, tier_val = _load_license_status(settings)
        assert status == ApplicationStatus.GATED_ACCESS
        assert ee_on is False


class TestSettingsEndpoint:
    """Integration tests via TestClient."""

    def _make_client(self, tmp_path, *, enforcement: bool = False):
        from fastapi.testclient import TestClient

        from src.configs import AppSettings, AuthSettings
        from src.db import AgenticSearchStore
        from src.servers.web.app import SearchExperienceSettings, create_web_app

        store = AgenticSearchStore(tmp_path / "db.sqlite3")
        settings = AppSettings(
            auth=AuthSettings(super_users=("admin",)),
            license_enforcement_enabled=enforcement,
        )
        app = create_web_app(
            SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"),
            app_settings=settings,
            store=store,
        )
        return TestClient(app), store

    def test_settings_endpoint_is_public(self, tmp_path) -> None:
        client, store = self._make_client(tmp_path)
        resp = client.get("/settings")
        assert resp.status_code == 200
        store.close()

    def test_settings_returns_expected_fields(self, tmp_path) -> None:
        client, store = self._make_client(tmp_path)
        resp = client.get("/settings")
        data = resp.json()
        assert "ee_features_enabled" in data
        assert "tier" in data
        assert "license_enforcement_enabled" in data
        store.close()

    def test_enforcement_disabled_enables_ee_features(self, tmp_path) -> None:
        client, store = self._make_client(tmp_path, enforcement=False)
        resp = client.get("/settings")
        data = resp.json()
        assert data["ee_features_enabled"] is True
        assert data["tier"] == Tier.ENTERPRISE.value
        store.close()

    def test_enforcement_enabled_without_license_disables_ee(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(tmp_path))
        client, store = self._make_client(tmp_path, enforcement=True)
        resp = client.get("/settings")
        data = resp.json()
        assert data["ee_features_enabled"] is False
        assert data["application_status"] == ApplicationStatus.GATED_ACCESS.value
        store.close()


class TestSettingsDefaults:
    def test_default_ee_features_disabled(self) -> None:
        s = Settings()
        assert s.ee_features_enabled is False
        assert s.license_enforcement_enabled is False
