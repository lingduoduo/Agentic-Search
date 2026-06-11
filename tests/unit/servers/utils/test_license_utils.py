"""Tests for license signature verification (src/utils/license.py).

Covers verify_license_signature with real RSA key pairs — the existing tests
in test_utils.py cover the status / valid helpers, so this file focuses on
the signature verification path which requires cryptography.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from unittest.mock import patch

import pytest

pytest.importorskip("cryptography", reason="cryptography package required")

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa

from src.internal.utils.license import LicensePayload
from src.internal.utils.license import get_license_status
from src.internal.utils.license import is_license_valid
from src.internal.utils.license import verify_license_signature


def _generate_key_pair():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private, private.public_key()


def _make_license(private_key, expires_days: int = 365, max_seats: int = 10) -> str:
    payload = LicensePayload(
        expires_at=datetime.now(timezone.utc) + timedelta(days=expires_days),
        max_seats=max_seats,
        features=["enterprise"],
    )
    payload_json = json.dumps(payload.model_dump(mode="json"), sort_keys=True)
    sig = private_key.sign(
        payload_json.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(
        json.dumps(
            {
                "payload": payload.model_dump(mode="json"),
                "signature": base64.b64encode(sig).decode(),
            }
        ).encode()
    ).decode()


class TestVerifyLicenseSignature:
    def test_valid_signature_returns_payload(self) -> None:
        private, public = _generate_key_pair()
        license_data = _make_license(private)
        with patch("src.internal.utils.license._get_public_key", return_value=public):
            result = verify_license_signature(license_data)
        assert result.max_seats == 10
        assert "enterprise" in result.features

    def test_wrong_public_key_raises(self) -> None:
        private, _ = _generate_key_pair()
        _, other_public = _generate_key_pair()
        license_data = _make_license(private)
        with patch(
            "src.internal.utils.license._get_public_key", return_value=other_public
        ):
            with pytest.raises(ValueError, match="[Ii]nvalid"):
                verify_license_signature(license_data)

    def test_tampered_payload_raises(self) -> None:
        private, public = _generate_key_pair()
        # Build a license but swap max_seats to a different value
        payload = LicensePayload(
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
            max_seats=10,
        )
        payload_json = json.dumps(payload.model_dump(mode="json"), sort_keys=True)
        sig = private.sign(
            payload_json.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )
        tampered = payload.model_dump(mode="json")
        tampered["max_seats"] = 9999
        encoded = base64.b64encode(
            json.dumps(
                {"payload": tampered, "signature": base64.b64encode(sig).decode()}
            ).encode()
        ).decode()
        with patch("src.internal.utils.license._get_public_key", return_value=public):
            with pytest.raises(ValueError):
                verify_license_signature(encoded)

    def test_invalid_base64_raises(self) -> None:
        with pytest.raises(ValueError):
            verify_license_signature("not-valid-base64!!!")

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError):
            verify_license_signature(base64.b64encode(b"not json").decode())


class TestGetLicenseStatus:
    def test_active_license(self) -> None:
        payload = LicensePayload(
            expires_at=datetime.now(timezone.utc) + timedelta(days=30)
        )
        from src.internal.utils.license import ApplicationStatus

        assert get_license_status(payload) == ApplicationStatus.ACTIVE

    def test_expired_no_grace(self) -> None:
        from src.internal.utils.license import ApplicationStatus

        payload = LicensePayload(
            expires_at=datetime.now(timezone.utc) - timedelta(days=1)
        )
        assert get_license_status(payload) == ApplicationStatus.GATED_ACCESS

    def test_within_grace_period(self) -> None:
        from src.internal.utils.license import ApplicationStatus
        from src.internal.utils.license_expiry import get_grace_period_end

        payload = LicensePayload(
            expires_at=datetime.now(timezone.utc) - timedelta(days=1)
        )
        grace = get_grace_period_end(payload.expires_at)
        assert get_license_status(payload, grace) == ApplicationStatus.GRACE_PERIOD

    def test_grace_period_expired_returns_gated(self) -> None:
        from src.internal.utils.license import ApplicationStatus

        payload = LicensePayload(
            expires_at=datetime.now(timezone.utc) - timedelta(days=30)
        )
        grace = datetime.now(timezone.utc) - timedelta(days=1)
        assert get_license_status(payload, grace) == ApplicationStatus.GATED_ACCESS


class TestIsLicenseValid:
    def test_valid_not_expired(self) -> None:
        payload = LicensePayload(
            expires_at=datetime.now(timezone.utc) + timedelta(days=1)
        )
        assert is_license_valid(payload) is True

    def test_expired(self) -> None:
        payload = LicensePayload(
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)
        )
        assert is_license_valid(payload) is False
