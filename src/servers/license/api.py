"""License API for self-hosted deployments.

Provides endpoints for uploading, verifying, refreshing, and deleting
the RSA-signed license. The sampled Onyx version used SQLAlchemy ORM
(ee.onyx.db.license) and Redis for caching. This repo stores the license
as a plain file in AGENTIC_SEARCH_DATA_DIR alongside enterprise settings.

Endpoints:
  GET    /license          — current license status
  GET    /license/seats    — seat usage
  POST   /license/upload   — upload signed .lic file (air-gapped installs)
  POST   /license/claim    — claim from cloud data plane (online installs)
  POST   /license/refresh  — re-verify from stored file
  DELETE /license          — delete stored license
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from datetime import timezone
from pathlib import Path

import requests
from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile

from src.auth import AuthenticatedUser
from src.configs import AppSettings
from src.servers.license.models import LicenseResponse
from src.servers.license.models import LicenseSource
from src.servers.license.models import LicenseStatusResponse
from src.servers.license.models import LicenseUploadResponse
from src.servers.license.models import SeatUsageResponse
from src.utils.license import LicensePayload
from src.utils.license import get_license_status
from src.utils.license import verify_license_signature
from src.utils.license_expiry import get_expiry_warning_stage
from src.utils.license_expiry import get_grace_period_end
from src.servers._auth import make_require_admin

logger = logging.getLogger(__name__)

_PEM_BEGIN = "-----BEGIN AGENTIC SEARCH LICENSE-----"
_PEM_END = "-----END AGENTIC SEARCH LICENSE-----"


# ---------------------------------------------------------------------------
# File-backed license store
# ---------------------------------------------------------------------------


def _license_dir() -> Path:
    path = Path(
        os.environ.get(
            "AGENTIC_SEARCH_DATA_DIR",
            Path.home() / ".local/share/agentic_search",
        )
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _license_dat_path() -> Path:
    return _license_dir() / "license.dat"


def _license_meta_path() -> Path:
    return _license_dir() / "license_meta.json"


def _save_license(license_data: str, source: LicenseSource) -> None:
    _license_dat_path().write_text(license_data)
    meta = {"source": source.value, "stored_at": datetime.now(timezone.utc).isoformat()}
    _license_meta_path().write_text(json.dumps(meta))


def _load_license_data() -> str | None:
    path = _license_dat_path()
    return path.read_text().strip() if path.exists() else None


def _load_license_source() -> LicenseSource | None:
    path = _license_meta_path()
    if not path.exists():
        return None
    try:
        return LicenseSource(json.loads(path.read_text()).get("source"))
    except (ValueError, KeyError):
        return None


def _delete_license_files() -> bool:
    dat = _license_dat_path()
    meta = _license_meta_path()
    deleted = dat.exists()
    dat.unlink(missing_ok=True)
    meta.unlink(missing_ok=True)
    return deleted


def _strip_pem(content: str) -> str:
    content = content.strip()
    if content.startswith(_PEM_BEGIN) and content.endswith(_PEM_END):
        lines = content.split("\n")
        return "\n".join(lines[1:-1]).strip()
    return content


def _status_from_payload(
    payload: LicensePayload,
    source: LicenseSource | None,
) -> LicenseStatusResponse:
    grace_end = get_grace_period_end(payload.expires_at)
    app_status = get_license_status(payload, grace_end)
    stage = get_expiry_warning_stage(payload.expires_at)
    return LicenseStatusResponse(
        has_license=True,
        max_seats=payload.max_seats,
        tenant_id=payload.tenant_id,
        expires_at=payload.expires_at,
        grace_period_end=grace_end,
        status=app_status,
        expiry_warning_stage=stage,
        source=source,
        features=payload.features,
    )


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_license_router(app_settings: AppSettings) -> APIRouter:
    """Return an APIRouter for license management endpoints."""

    router = APIRouter(prefix="/license", tags=["license"])

    _require_admin = make_require_admin(app_settings)

    def _load_and_verify() -> LicensePayload | None:
        data = _load_license_data()
        if not data:
            return None
        try:
            return verify_license_signature(data)
        except ValueError:
            return None

    @router.get("")
    def get_license_status_endpoint(
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> LicenseStatusResponse:
        """Return current license status."""
        payload = _load_and_verify()
        if payload is None:
            return LicenseStatusResponse(has_license=False)
        return _status_from_payload(payload, _load_license_source())

    @router.get("/seats")
    def get_seat_usage(
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> SeatUsageResponse:
        """Return seat limit from the stored license (no active user count)."""
        payload = _load_and_verify()
        if payload is None or payload.max_seats is None:
            return SeatUsageResponse(total_seats=0, used_seats=0, available_seats=0)
        return SeatUsageResponse(
            total_seats=payload.max_seats,
            used_seats=0,
            available_seats=payload.max_seats,
        )

    @router.post("/upload")
    async def upload_license(
        license_file: UploadFile = File(...),
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> LicenseUploadResponse:
        """Upload a signed license file (.lic or base64 string)."""
        try:
            raw = await license_file.read()
            license_data = _strip_pem(raw.decode("utf-8").strip())
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400, detail="Invalid license file encoding."
            ) from exc

        try:
            payload = verify_license_signature(license_data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        _save_license(license_data, LicenseSource.MANUAL_UPLOAD)
        return LicenseUploadResponse(
            success=True,
            message=(
                f"License stored. Expires {payload.expires_at.date()}."
                + (f" Max seats: {payload.max_seats}." if payload.max_seats else "")
            ),
        )

    @router.post("/claim")
    async def claim_license(
        session_id: str | None = None,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> LicenseResponse:
        """Claim or refresh license from the cloud data plane.

        Requires AppSettings.cloud_data_plane_url to be configured.
        """
        base_url = app_settings.cloud_data_plane_url
        if not base_url:
            raise HTTPException(
                status_code=501,
                detail=(
                    "cloud_data_plane_url is not configured. "
                    "Set AGENTIC_SEARCH_CLOUD_DATA_PLANE_URL or upload the license manually."
                ),
            )

        try:
            if session_id:
                url = f"{base_url.rstrip('/')}/proxy/claim-license"
                resp = requests.post(
                    url,
                    json={"session_id": session_id},
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
            else:
                existing = _load_license_data()
                if not existing:
                    raise HTTPException(
                        status_code=400,
                        detail="No stored license. Provide session_id after checkout.",
                    )
                payload_existing = verify_license_signature(existing)
                url = (
                    f"{base_url.rstrip('/')}/proxy/license/{payload_existing.tenant_id}"
                )
                resp = requests.get(
                    url,
                    headers={"Authorization": f"Bearer {existing}"},
                    timeout=30,
                )

            resp.raise_for_status()
            license_data = resp.json().get("license")
            if not license_data:
                raise HTTPException(status_code=502, detail="No license in response.")

            payload = verify_license_signature(license_data)
            _save_license(license_data, LicenseSource.AUTO_FETCH)
            return LicenseResponse(
                success=True,
                expires_at=payload.expires_at,
                max_seats=payload.max_seats,
                tenant_id=payload.tenant_id,
            )

        except HTTPException:
            raise
        except requests.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502, detail=f"Could not reach license server: {exc}"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/refresh")
    async def refresh_license(
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> LicenseStatusResponse:
        """Re-verify the stored license and return updated status."""
        payload = _load_and_verify()
        if payload is None:
            return LicenseStatusResponse(has_license=False)
        return _status_from_payload(payload, _load_license_source())

    @router.delete("")
    async def delete_license_endpoint(
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> dict[str, bool]:
        """Delete the stored license."""
        deleted = _delete_license_files()
        return {"deleted": deleted}

    return router


__all__ = ["create_license_router"]
