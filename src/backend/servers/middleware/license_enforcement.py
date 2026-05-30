"""License enforcement middleware for self-hosted deployments.

The sampled Onyx version used Redis (get_cached_license_metadata),
SQLAlchemy, and CACHE_TRANSIENT_ERRORS. This repo uses the file-backed
license store from src/servers/license/api.py and the allowlist from
src/configs/license_enforcement_config.py.

Behaviour (when license_enforcement_enabled=True in AppSettings):
  1. GATED_ACCESS (license fully expired) → block all non-exempt paths.
  2. No license stored → pass (community tier; feature gating handled by
     tier_gate middleware).
  3. Valid or grace-period license → pass.

Replaces the minimal _LicenseMiddleware in src/servers/web/app.py.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from collections.abc import Callable

from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
from fastapi.responses import JSONResponse

from src.backend.configs import AppSettings
from src.backend.configs.license_enforcement_config import is_license_enforcement_exempt
from src.backend.utils.license import ApplicationStatus
from src.backend.utils.license import get_license_status
from src.backend.utils.license import verify_license_signature
from src.backend.utils.license_expiry import get_grace_period_end

logger = logging.getLogger(__name__)


def _load_payload():
    """Load and verify the stored license, returning None if absent or invalid."""
    try:
        from src.backend.servers.license.api import _load_license_data

        data = _load_license_data()
        if not data:
            return None
        return verify_license_signature(data)
    except Exception:
        return None


def add_license_enforcement_middleware(
    app: FastAPI,
    app_settings: AppSettings,
) -> None:
    """Register the license enforcement middleware on *app*.

    No-ops immediately when ``app_settings.license_enforcement_enabled``
    is False so local dev and legacy deployments are unaffected.
    """
    if not app_settings.license_enforcement_enabled:
        logger.debug("License enforcement middleware disabled.")
        return

    logger.info("License enforcement middleware registered.")

    @app.middleware("http")
    async def enforce_license(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path
        if path.startswith("/api/"):
            path = path[4:]

        if is_license_enforcement_exempt(path):
            return await call_next(request)

        payload = _load_payload()

        if payload is None:
            # No license — community tier paths stay accessible.
            return await call_next(request)

        grace_end = get_grace_period_end(payload.expires_at)
        status = get_license_status(payload, grace_end)

        if status == ApplicationStatus.GATED_ACCESS:
            logger.info("[license_enforcement] Blocking gated path: %s", path)
            return JSONResponse(
                status_code=402,
                content={
                    "detail": {
                        "error": "license_expired",
                        "message": "Your license has expired. Please renew.",
                    }
                },
            )

        return await call_next(request)


__all__ = ["add_license_enforcement_middleware"]
