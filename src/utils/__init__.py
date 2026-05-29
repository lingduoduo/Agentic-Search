"""Utility helpers for Agentic Search.

Exports are loaded lazily so importing one utility module does not require every
optional dependency used by the package.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "decrypt_bytes_to_string": (".encryption", "decrypt_bytes_to_string"),
    "encrypt_string_to_bytes": (".encryption", "encrypt_string_to_bytes"),
    "verify_encryption": (".encryption", "verify_encryption"),
    "ApplicationStatus": (".license", "ApplicationStatus"),
    "LicenseData": (".license", "LicenseData"),
    "LicensePayload": (".license", "LicensePayload"),
    "get_license_status": (".license", "get_license_status"),
    "is_license_valid": (".license", "is_license_valid"),
    "verify_license_signature": (".license", "verify_license_signature"),
    "ExpiryWarningStage": (".license_expiry", "ExpiryWarningStage"),
    "get_expiry_warning_stage": (".license_expiry", "get_expiry_warning_stage"),
    "get_grace_days_remaining": (".license_expiry", "get_grace_days_remaining"),
    "get_grace_period_end": (".license_expiry", "get_grace_period_end"),
    "notify_admins_for_stage": (".license_notifications", "notify_admins_for_stage"),
    "alias_user": (".posthog_client", "alias_user"),
    "build_posthog_client": (".posthog_client", "build_posthog_client"),
    "get_anon_id_from_request": (".posthog_client", "get_anon_id_from_request"),
    "parse_posthog_cookie": (".posthog_client", "parse_posthog_cookie"),
    "event_telemetry": (".telemetry", "event_telemetry"),
    "identify_user": (".telemetry", "identify_user"),
    "get_tier": (".tier", "get_tier"),
    "tier_at_least": (".tier", "tier_at_least"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
