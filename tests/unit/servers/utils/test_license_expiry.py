"""Parametrized boundary tests for license expiry stages.

More thorough than the tests in test_utils.py — covers every boundary point
at day-precision using a pinned "now" to keep assertions stable.
"""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from unittest.mock import patch

import pytest

from src.internal.utils.license_expiry import (
    LICENSE_GRACE_PERIOD_DAYS,
    ExpiryWarningStage,
    get_expiry_warning_stage,
    get_grace_days_remaining,
    get_grace_period_end,
)

NOW = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "delta, expected",
    [
        (timedelta(days=60), ExpiryWarningStage.NONE),
        (timedelta(days=31), ExpiryWarningStage.NONE),
        (timedelta(days=30), ExpiryWarningStage.T_30D),
        (timedelta(days=15), ExpiryWarningStage.T_30D),
        (timedelta(days=14, seconds=1), ExpiryWarningStage.T_30D),
        (timedelta(days=14), ExpiryWarningStage.T_14D),
        (timedelta(days=2), ExpiryWarningStage.T_14D),
        (timedelta(days=1, seconds=1), ExpiryWarningStage.T_14D),
        (timedelta(days=1), ExpiryWarningStage.T_1D),
        (timedelta(hours=12), ExpiryWarningStage.T_1D),
        (timedelta(seconds=1), ExpiryWarningStage.T_1D),
        (timedelta(0), ExpiryWarningStage.GRACE),
        (timedelta(hours=-1), ExpiryWarningStage.GRACE),
        (timedelta(days=-1), ExpiryWarningStage.GRACE),
        (timedelta(days=-(LICENSE_GRACE_PERIOD_DAYS - 1)), ExpiryWarningStage.GRACE),
        (timedelta(days=-LICENSE_GRACE_PERIOD_DAYS), ExpiryWarningStage.NONE),
        (timedelta(days=-30), ExpiryWarningStage.NONE),
    ],
)
def test_get_expiry_warning_stage_boundaries(
    delta: timedelta, expected: ExpiryWarningStage
) -> None:
    with patch("src.internal.utils.license_expiry.datetime") as mock_dt:
        mock_dt.now.return_value = NOW
        assert get_expiry_warning_stage(NOW + delta) == expected


def test_grace_days_remaining_full_window() -> None:
    just_expired = NOW - timedelta(seconds=1)
    with patch("src.internal.utils.license_expiry.datetime") as mock_dt:
        mock_dt.now.return_value = NOW
        assert get_grace_days_remaining(just_expired) == LICENSE_GRACE_PERIOD_DAYS


def test_grace_days_remaining_one_day_left() -> None:
    expires = NOW - timedelta(days=LICENSE_GRACE_PERIOD_DAYS - 1)
    with patch("src.internal.utils.license_expiry.datetime") as mock_dt:
        mock_dt.now.return_value = NOW
        assert get_grace_days_remaining(expires) == 1


def test_grace_days_remaining_exhausted() -> None:
    expires = NOW - timedelta(days=LICENSE_GRACE_PERIOD_DAYS)
    with patch("src.internal.utils.license_expiry.datetime") as mock_dt:
        mock_dt.now.return_value = NOW
        assert get_grace_days_remaining(expires) == 0


def test_get_grace_period_end_is_expires_plus_grace_window() -> None:
    expires = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert get_grace_period_end(expires) == expires + timedelta(
        days=LICENSE_GRACE_PERIOD_DAYS
    )
