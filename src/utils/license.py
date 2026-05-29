"""RSA-4096 license signature verification utilities."""

from __future__ import annotations

import base64
import json
import logging
import os
from datetime import datetime
from datetime import timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

try:
    from cryptography.exceptions import InvalidSignature
except ModuleNotFoundError:

    class InvalidSignature(Exception):  # type: ignore[no-redef]
        pass


logger = logging.getLogger(__name__)

_LICENSE_PUBLIC_KEY_PATH = (
    Path(__file__).parent.parent.parent / "keys" / "license_public_key.pem"
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ApplicationStatus(str, Enum):
    ACTIVE = "active"
    GRACE_PERIOD = "grace_period"
    GATED_ACCESS = "gated_access"


class LicensePayload(BaseModel):
    """Verified fields extracted from a signed license token."""

    expires_at: datetime
    tenant_id: str | None = None
    max_seats: int | None = None
    features: list[str] = []


class LicenseData(BaseModel):
    """Raw wire format: a Pydantic-validated payload plus a base64 signature."""

    payload: LicensePayload
    signature: str


# ---------------------------------------------------------------------------
# Key loading
# ---------------------------------------------------------------------------


def _get_public_key() -> Any:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
    except ModuleNotFoundError as exc:
        raise ValueError(
            "License signature verification requires the cryptography package."
        ) from exc

    key_pem = os.environ.get("LICENSE_PUBLIC_KEY_PEM")
    if not key_pem:
        if not _LICENSE_PUBLIC_KEY_PATH.exists():
            raise ValueError(
                f"License public key not found at {_LICENSE_PUBLIC_KEY_PATH}. "
                "Set the LICENSE_PUBLIC_KEY_PEM env var or place the key file there."
            )
        key_pem = _LICENSE_PUBLIC_KEY_PATH.read_text()
    key = serialization.load_pem_public_key(key_pem.encode())
    if not isinstance(key, RSAPublicKey):
        raise ValueError("Expected an RSA public key.")
    return key


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_license_signature(license_data: str) -> LicensePayload:
    """Verify an RSA-4096/PSS-SHA256 signature and return the payload.

    ``license_data`` must be a base64-encoded JSON string containing
    ``payload`` (dict) and ``signature`` (base64 string).

    Raises ``ValueError`` on any validation or signature failure.
    """
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding as asym_padding

        decoded: dict[str, Any] = json.loads(base64.b64decode(license_data))
        license_obj = LicenseData(**decoded)

        # Use the ORIGINAL payload dict for signature verification — not the
        # Pydantic-serialised version, which may format datetimes differently
        # (e.g. "+00:00" vs "Z") and would break an otherwise valid signature.
        original_payload = decoded.get("payload", {})
        payload_json = json.dumps(original_payload, sort_keys=True)
        signature_bytes = base64.b64decode(license_obj.signature)

        _get_public_key().verify(
            signature_bytes,
            payload_json.encode(),
            asym_padding.PSS(
                mgf=asym_padding.MGF1(hashes.SHA256()),
                salt_length=asym_padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return license_obj.payload

    except InvalidSignature:
        logger.error("[verify_license] FAILED: invalid signature")
        raise ValueError("Invalid license signature.")
    except json.JSONDecodeError as exc:
        logger.error("[verify_license] FAILED: JSON error: %s", exc)
        raise ValueError("Invalid license format: not valid JSON.")
    except (ValueError, KeyError, TypeError) as exc:
        logger.error("[verify_license] FAILED: %s: %s", type(exc).__name__, exc)
        raise ValueError(f"Invalid license format: {exc}") from exc
    except Exception:
        logger.exception("[verify_license] FAILED: unexpected error")
        raise ValueError("License verification failed.")


# ---------------------------------------------------------------------------
# Status derivation
# ---------------------------------------------------------------------------


def get_license_status(
    payload: LicensePayload,
    grace_period_end: datetime | None = None,
) -> ApplicationStatus:
    """Return the current ``ApplicationStatus`` based on expiry and grace period."""
    now = datetime.now(timezone.utc)
    if grace_period_end and now > grace_period_end:
        return ApplicationStatus.GATED_ACCESS
    if now > payload.expires_at:
        if grace_period_end and now <= grace_period_end:
            return ApplicationStatus.GRACE_PERIOD
        return ApplicationStatus.GATED_ACCESS
    return ApplicationStatus.ACTIVE


def is_license_valid(payload: LicensePayload) -> bool:
    """Return ``True`` if the license has not expired."""
    return datetime.now(timezone.utc) <= payload.expires_at


__all__ = [
    "ApplicationStatus",
    "LicenseData",
    "LicensePayload",
    "get_license_status",
    "is_license_valid",
    "verify_license_signature",
]
