"""License-expiry tiered notification copy and orchestration.

This module generates human-readable notification copy (banner title,
description, email subject) for each expiry warning stage. The repo has
no email or in-app notification infrastructure, so ``notify_admins_for_stage``
logs the notification at the appropriate level and returns the copy strings so
callers can surface them through whatever channel they support.
"""

from __future__ import annotations

import logging
from datetime import date
from datetime import datetime
from datetime import timezone
from typing import Any

from src.internal.utils.license_expiry import ExpiryWarningStage
from src.internal.utils.license_expiry import get_grace_days_remaining

logger = logging.getLogger(__name__)


def _build_copy(
    stage: ExpiryWarningStage,
    expires_at: datetime,
    grace_days_remaining: int,
) -> tuple[str, str, str]:
    """Return ``(banner_title, banner_description, email_subject)`` for *stage*."""
    expires_str = expires_at.strftime("%Y-%m-%d")
    if stage == ExpiryWarningStage.T_30D:
        return (
            f"License expires {expires_str}",
            "Your license will expire in approximately 30 days. "
            "Renew to avoid service interruption.",
            "Action required: license expires in ~30 days",
        )
    if stage == ExpiryWarningStage.T_14D:
        return (
            f"License expires {expires_str}",
            "Your license will expire in approximately 2 weeks. "
            "Renewal must be completed soon to avoid service interruption.",
            "Action required: license expires in ~2 weeks",
        )
    if stage == ExpiryWarningStage.T_1D:
        return (
            f"License expires tomorrow ({expires_str})",
            "Your license expires within 24 hours. "
            "Renew immediately to avoid service interruption.",
            "URGENT: license expires within 24 hours",
        )
    if stage == ExpiryWarningStage.GRACE:
        return (
            f"License expired — {grace_days_remaining} grace days remaining",
            f"Your license expired on {expires_str}. You have "
            f"{grace_days_remaining} day(s) of grace access remaining. Renew now.",
            f"License expired — {grace_days_remaining} grace days remaining",
        )
    raise ValueError(f"Unsupported stage for notification copy: {stage!r}")


def _build_additional_data(
    stage: ExpiryWarningStage,
    expires_at: datetime,
    today: date,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "stage": stage.value,
        "expires_at": expires_at.isoformat(),
    }
    if stage == ExpiryWarningStage.GRACE:
        data["sent_date"] = today.isoformat()
    return data


def notify_admins_for_stage(
    stage: ExpiryWarningStage,
    expires_at: datetime,
    admin_emails: list[str] | None = None,
) -> tuple[str, str, str]:
    """Log and return notification copy for the given expiry stage.

    ``admin_emails`` is accepted for callers that want to trigger their own
    email delivery; this module itself does not send email.

    Returns ``(banner_title, banner_description, email_subject)`` so callers
    can display the copy in-app. Returns empty strings for ``NONE`` stage.
    """
    if stage == ExpiryWarningStage.NONE:
        return "", "", ""

    today = datetime.now(timezone.utc).date()
    grace_days = get_grace_days_remaining(expires_at)
    title, description, email_subject = _build_copy(stage, expires_at, grace_days)
    additional = _build_additional_data(stage, expires_at, today)

    log_level = logging.WARNING if stage == ExpiryWarningStage.GRACE else logging.INFO
    logger.log(
        log_level,
        "License expiry notification: stage=%s expires_at=%s grace_days=%d",
        stage.value,
        expires_at.isoformat(),
        grace_days,
    )

    if admin_emails:
        for email in admin_emails:
            logger.info(
                "License notification for %s: subject=%r additional=%r",
                email,
                email_subject,
                additional,
            )

    return title, description, email_subject


__all__ = [
    "ExpiryWarningStage",
    "notify_admins_for_stage",
]
