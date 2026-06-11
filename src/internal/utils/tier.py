"""Tier resolution for Agentic Search instances.

Simplified from the sampled multi-tenant/Redis/CP version: this repo is a
single-tenant self-hosted deployment. Tier is derived from
``AppSettings.license_enforcement_enabled`` and an optional verified
``LicensePayload``.

  enforcement disabled → ENTERPRISE (full access for dev / legacy mode)
  enforcement enabled, valid license → tier from license payload features
  enforcement enabled, no license    → FREE
"""

from __future__ import annotations

from src.internal.configs import AppSettings
from src.internal.configs import Tier
from src.internal.configs import load_app_settings


def get_tier(
    app_settings: AppSettings | None = None,
    license_payload: object | None = None,
) -> Tier:
    """Return the effective ``Tier`` for the current deployment.

    Args:
        app_settings: Loaded settings. ``load_app_settings()`` is used when
            ``None``.
        license_payload: Optional ``LicensePayload`` from
            ``src.utils.license.verify_license_signature``. When present its
            ``features`` list is consulted to determine BUSINESS vs ENTERPRISE.

    Returns:
        ``Tier.ENTERPRISE`` when enforcement is off (dev / legacy EE mode).
        ``Tier.FREE`` when enforcement is on but no valid license is supplied.
        ``Tier.BUSINESS`` or ``Tier.ENTERPRISE`` based on the license payload.
    """
    settings = app_settings or load_app_settings()

    if not settings.license_enforcement_enabled:
        return Tier.ENTERPRISE

    if license_payload is None:
        return Tier.FREE

    features: list[str] = getattr(license_payload, "features", []) or []
    if "enterprise" in [f.lower() for f in features]:
        return Tier.ENTERPRISE
    return Tier.BUSINESS


def tier_at_least(current: Tier, required: Tier) -> bool:
    """Return ``True`` if *current* is at least *required*."""
    _RANK = {Tier.FREE: 0, Tier.BUSINESS: 1, Tier.ENTERPRISE: 2}
    return _RANK[current] >= _RANK[required]


__all__ = [
    "get_tier",
    "tier_at_least",
]
