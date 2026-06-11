"""Tests for tier resolution (src/utils/tier.py).

Our tier function is simpler than the sampled version: no Redis, no DB.
It reads AppSettings.license_enforcement_enabled and an optional LicensePayload.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.internal.configs import AppSettings, Tier
from src.internal.utils.license import LicensePayload
from src.internal.utils.tier import get_tier, tier_at_least


def _payload(features: list[str], days: int = 365) -> LicensePayload:
    return LicensePayload(
        expires_at=datetime.now(timezone.utc) + timedelta(days=days),
        features=features,
    )


class TestGetTier:
    def test_enforcement_disabled_returns_enterprise(self) -> None:
        settings = AppSettings(license_enforcement_enabled=False)
        assert get_tier(settings) == Tier.ENTERPRISE

    def test_enforcement_enabled_no_payload_returns_free(self) -> None:
        settings = AppSettings(license_enforcement_enabled=True)
        assert get_tier(settings) == Tier.FREE

    def test_enterprise_feature_in_payload_returns_enterprise(self) -> None:
        settings = AppSettings(license_enforcement_enabled=True)
        payload = _payload(["enterprise"])
        assert get_tier(settings, payload) == Tier.ENTERPRISE

    def test_enterprise_feature_case_insensitive(self) -> None:
        settings = AppSettings(license_enforcement_enabled=True)
        payload = _payload(["ENTERPRISE"])
        assert get_tier(settings, payload) == Tier.ENTERPRISE

    def test_no_enterprise_feature_returns_business(self) -> None:
        settings = AppSettings(license_enforcement_enabled=True)
        payload = _payload(["analytics", "hooks"])
        assert get_tier(settings, payload) == Tier.BUSINESS

    def test_empty_features_returns_business(self) -> None:
        settings = AppSettings(license_enforcement_enabled=True)
        payload = _payload([])
        assert get_tier(settings, payload) == Tier.BUSINESS

    def test_uses_load_app_settings_when_none(self) -> None:
        # Should not raise even without explicit settings
        result = get_tier()
        assert isinstance(result, Tier)


class TestTierAtLeast:
    @pytest.mark.parametrize(
        "current, required, expected",
        [
            (Tier.FREE, Tier.FREE, True),
            (Tier.FREE, Tier.BUSINESS, False),
            (Tier.FREE, Tier.ENTERPRISE, False),
            (Tier.BUSINESS, Tier.FREE, True),
            (Tier.BUSINESS, Tier.BUSINESS, True),
            (Tier.BUSINESS, Tier.ENTERPRISE, False),
            (Tier.ENTERPRISE, Tier.FREE, True),
            (Tier.ENTERPRISE, Tier.BUSINESS, True),
            (Tier.ENTERPRISE, Tier.ENTERPRISE, True),
        ],
    )
    def test_tier_ordering(self, current: Tier, required: Tier, expected: bool) -> None:
        assert tier_at_least(current, required) is expected
