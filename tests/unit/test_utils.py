"""Tests for src/utils — encryption, license expiry, notifications, posthog, tier."""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import pytest

from src.backend.configs import AppSettings
from src.backend.configs import Tier
from src.backend.utils.encryption import decrypt_bytes_to_string
from src.backend.utils.encryption import encrypt_string_to_bytes
from src.backend.utils.encryption import verify_encryption
from src.backend.utils.license import ApplicationStatus
from src.backend.utils.license import LicensePayload
from src.backend.utils.license import get_license_status
from src.backend.utils.license import is_license_valid
from src.backend.utils.license_expiry import ExpiryWarningStage
from src.backend.utils.license_expiry import get_expiry_warning_stage
from src.backend.utils.license_expiry import get_grace_days_remaining
from src.backend.utils.license_expiry import get_grace_period_end
from src.backend.utils.license_notifications import notify_admins_for_stage
from src.backend.utils.posthog_client import build_posthog_client
from src.backend.utils.posthog_client import parse_posthog_cookie
from src.backend.utils.telemetry import event_telemetry
from src.backend.utils.telemetry import identify_user
from src.backend.utils.tier import get_tier
from src.backend.utils.tier import tier_at_least

_NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# encryption
# ---------------------------------------------------------------------------


def test_encryption_passthrough_when_no_key(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_ENCRYPTION_KEY", raising=False)
    from src.backend.utils.encryption import _get_trimmed_key

    _get_trimmed_key.cache_clear()
    text = "hello world"
    assert encrypt_string_to_bytes(text) == text.encode()
    assert decrypt_bytes_to_string(text.encode()) == text


def test_encryption_roundtrip_with_explicit_key():
    key = "thisis16byteskey"
    text = "sensitive-credential-value"
    encrypted = encrypt_string_to_bytes(text, key=key)
    assert encrypted != text.encode()
    assert decrypt_bytes_to_string(encrypted, key=key) == text


def test_encryption_roundtrip_uses_env_key(monkeypatch):
    key = "32-byte-key-for-aes-256-bit-enc!"
    monkeypatch.setenv("AGENTIC_SEARCH_ENCRYPTION_KEY", key)
    from src.backend.utils.encryption import _get_trimmed_key

    _get_trimmed_key.cache_clear()
    text = "my secret"
    encrypted = encrypt_string_to_bytes(text)
    assert decrypt_bytes_to_string(encrypted) == text
    _get_trimmed_key.cache_clear()


def test_encryption_different_keys_produce_different_ciphertext():
    a = encrypt_string_to_bytes("same input", key="1234567890123456")
    b = encrypt_string_to_bytes("same input", key="abcdefghijklmnop")
    assert a != b


def test_encryption_rejects_short_key():
    from src.backend.utils.encryption import _get_trimmed_key

    _get_trimmed_key.cache_clear()
    with pytest.raises(ValueError, match="too short"):
        encrypt_string_to_bytes("x", key="tooshort")
    _get_trimmed_key.cache_clear()


def test_verify_encryption_self_test_passes(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_ENCRYPTION_KEY", raising=False)
    from src.backend.utils.encryption import _get_trimmed_key

    _get_trimmed_key.cache_clear()
    verify_encryption()


# ---------------------------------------------------------------------------
# license_expiry
# ---------------------------------------------------------------------------


def test_expiry_stage_none_when_far_future():
    assert (
        get_expiry_warning_stage(_NOW + timedelta(days=31)) == ExpiryWarningStage.NONE
    )


def test_expiry_stage_t30d():
    assert (
        get_expiry_warning_stage(_NOW + timedelta(days=20)) == ExpiryWarningStage.T_30D
    )


def test_expiry_stage_t14d():
    assert (
        get_expiry_warning_stage(_NOW + timedelta(days=7)) == ExpiryWarningStage.T_14D
    )


def test_expiry_stage_t1d():
    assert (
        get_expiry_warning_stage(_NOW + timedelta(hours=12)) == ExpiryWarningStage.T_1D
    )


def test_expiry_stage_grace():
    assert (
        get_expiry_warning_stage(_NOW - timedelta(days=3)) == ExpiryWarningStage.GRACE
    )


def test_expiry_stage_none_after_grace():
    assert (
        get_expiry_warning_stage(_NOW - timedelta(days=20)) == ExpiryWarningStage.NONE
    )


def test_grace_period_end_is_14_days_after_expiry():
    exp = _NOW + timedelta(days=1)
    grace = get_grace_period_end(exp)
    assert abs((grace - exp).days - 14) <= 1


def test_grace_days_remaining_decreases_after_expiry():
    expired_recently = _NOW - timedelta(days=3)
    days = get_grace_days_remaining(expired_recently)
    assert 0 < days <= 14


# ---------------------------------------------------------------------------
# license (ApplicationStatus, LicensePayload)
# ---------------------------------------------------------------------------


def _payload(expires_at: datetime) -> LicensePayload:
    return LicensePayload(expires_at=expires_at)


def test_is_license_valid_true_before_expiry():
    assert is_license_valid(_payload(_NOW + timedelta(days=10)))


def test_is_license_valid_false_after_expiry():
    assert not is_license_valid(_payload(_NOW - timedelta(seconds=1)))


def test_get_license_status_active():
    assert (
        get_license_status(_payload(_NOW + timedelta(days=10)))
        == ApplicationStatus.ACTIVE
    )


def test_get_license_status_gated_when_expired_no_grace():
    assert (
        get_license_status(_payload(_NOW - timedelta(days=1)))
        == ApplicationStatus.GATED_ACCESS
    )


def test_get_license_status_grace_period():
    expired = _NOW - timedelta(days=3)
    grace_end = _NOW + timedelta(days=5)
    assert (
        get_license_status(_payload(expired), grace_end)
        == ApplicationStatus.GRACE_PERIOD
    )


def test_get_license_status_gated_when_grace_expired():
    expired = _NOW - timedelta(days=20)
    grace_end = _NOW - timedelta(days=5)
    assert (
        get_license_status(_payload(expired), grace_end)
        == ApplicationStatus.GATED_ACCESS
    )


# ---------------------------------------------------------------------------
# license_notifications
# ---------------------------------------------------------------------------


def test_notify_returns_empty_for_none_stage():
    exp = _NOW + timedelta(days=60)
    assert notify_admins_for_stage(ExpiryWarningStage.NONE, exp) == ("", "", "")


def test_notify_returns_copy_for_t30d():
    exp = _NOW + timedelta(days=25)
    title, desc, subj = notify_admins_for_stage(ExpiryWarningStage.T_30D, exp)
    assert "30 days" in desc
    assert "expires" in subj.lower()


def test_notify_returns_copy_for_grace():
    exp = _NOW - timedelta(days=3)
    title, desc, subj = notify_admins_for_stage(ExpiryWarningStage.GRACE, exp)
    assert "grace" in title.lower()
    assert "grace" in subj.lower()


# ---------------------------------------------------------------------------
# posthog_client
# ---------------------------------------------------------------------------


def test_build_posthog_client_returns_none_when_no_key():
    assert build_posthog_client(None) is None
    assert build_posthog_client("") is None


def test_parse_posthog_cookie_returns_none_on_bad_json():
    assert parse_posthog_cookie("not-json") is None


def test_parse_posthog_cookie_returns_none_without_distinct_id():
    import json
    from urllib.parse import quote

    assert parse_posthog_cookie(quote(json.dumps({"other_key": "val"}))) is None


def test_parse_posthog_cookie_returns_dict_with_distinct_id():
    import json
    from urllib.parse import quote

    payload = {"distinct_id": "user-123", "featureFlags": {}}
    result = parse_posthog_cookie(quote(json.dumps(payload)))
    assert result is not None
    assert result["distinct_id"] == "user-123"


# ---------------------------------------------------------------------------
# telemetry
# ---------------------------------------------------------------------------


def test_event_telemetry_noop_when_client_is_none():
    event_telemetry(None, "user-1", "page_view")  # must not raise


def test_identify_user_noop_when_client_is_none():
    identify_user(None, "user-1")  # must not raise


def test_event_telemetry_calls_capture(monkeypatch):
    captured = []

    class FakeClient:
        def capture(self, distinct_id, event, props):
            captured.append((distinct_id, event, props))

        def flush(self):
            pass

    event_telemetry(FakeClient(), "u1", "search", {"query": "ml"})
    assert captured == [("u1", "search", {"query": "ml"})]


# ---------------------------------------------------------------------------
# tier
# ---------------------------------------------------------------------------


def test_get_tier_enterprise_when_enforcement_disabled():
    settings = AppSettings()
    assert get_tier(settings) == Tier.ENTERPRISE


def test_get_tier_free_when_enforcement_enabled_no_license():
    from src.backend.configs import load_app_settings

    settings = load_app_settings({"AGENTIC_SEARCH_LICENSE_ENFORCEMENT_ENABLED": "true"})
    assert get_tier(settings) == Tier.FREE


def test_get_tier_enterprise_from_license_feature():
    from src.backend.configs import load_app_settings

    settings = load_app_settings({"AGENTIC_SEARCH_LICENSE_ENFORCEMENT_ENABLED": "true"})
    payload = LicensePayload(
        expires_at=_NOW + timedelta(days=365), features=["enterprise"]
    )
    assert get_tier(settings, license_payload=payload) == Tier.ENTERPRISE


def test_get_tier_business_from_license_without_enterprise_feature():
    from src.backend.configs import load_app_settings

    settings = load_app_settings({"AGENTIC_SEARCH_LICENSE_ENFORCEMENT_ENABLED": "true"})
    payload = LicensePayload(
        expires_at=_NOW + timedelta(days=365), features=["analytics"]
    )
    assert get_tier(settings, license_payload=payload) == Tier.BUSINESS


def test_tier_at_least():
    assert tier_at_least(Tier.ENTERPRISE, Tier.BUSINESS)
    assert tier_at_least(Tier.BUSINESS, Tier.BUSINESS)
    assert not tier_at_least(Tier.FREE, Tier.BUSINESS)
    assert not tier_at_least(Tier.BUSINESS, Tier.ENTERPRISE)
