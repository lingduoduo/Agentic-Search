"""Tier-gating middleware.

The previous version used Redis-backed get_tier(), AppErrorCode, and
shared_configs tenant context vars. This repo uses:
  - src/configs/license_enforcement_config.py — PATH_PREFIX_MIN_TIER,
    is_license_enforcement_exempt (the allowlist)
  - src/utils/tier.py — get_tier(app_settings)
  - src/utils/tier.py — tier_at_least()

For every non-exempt request:
  1. Find the longest matching prefix in PATH_PREFIX_MIN_TIER.
  2. No match → pass.
  3. Resolve effective tier from AppSettings + stored license.
  4. If tier insufficient → 402 with required_tier in payload.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from collections.abc import Callable

from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
from fastapi.responses import JSONResponse

from src.internal.configs import AppSettings
from src.internal.configs import Tier
from src.internal.configs.license_enforcement_config import PATH_PREFIX_MIN_TIER
from src.internal.configs.license_enforcement_config import (
    is_license_enforcement_exempt,
)
from src.internal.utils.tier import get_tier
from src.internal.utils.tier import tier_at_least

logger = logging.getLogger(__name__)

# Sorted longest-first so more-specific prefixes match before shorter ones.
_SORTED_GATES: list[tuple[str, Tier]] = sorted(
    PATH_PREFIX_MIN_TIER.items(),
    key=lambda item: len(item[0]),
    reverse=True,
)


def _required_tier(path: str) -> Tier | None:
    for prefix, required in _SORTED_GATES:
        if path == prefix or path.startswith(f"{prefix}/"):
            return required
    return None


def add_tier_gate_middleware(
    app: FastAPI,
    app_settings: AppSettings,
) -> None:
    """Register the tier-gate middleware on *app*.

    No-ops when PATH_PREFIX_MIN_TIER is empty (no gated paths configured).
    """
    if not _SORTED_GATES:
        return

    logger.info("Tier-gate middleware registered (entries=%d).", len(_SORTED_GATES))

    @app.middleware("http")
    async def enforce_tier_gate(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path
        if path.startswith("/api/"):
            path = path[4:]

        if is_license_enforcement_exempt(path):
            return await call_next(request)

        required = _required_tier(path)
        if required is None:
            return await call_next(request)

        license_payload = None
        try:
            from src.internal.servers.license.api import _load_license_data
            from src.internal.utils.license import verify_license_signature

            data = _load_license_data()
            if data:
                license_payload = verify_license_signature(data)
        except Exception:
            pass

        current_tier = get_tier(app_settings, license_payload)

        if tier_at_least(current_tier, required):
            return await call_next(request)

        logger.info(
            "[tier_gate] Blocking %s: current=%s required=%s",
            path,
            current_tier.value,
            required.value,
        )
        return JSONResponse(
            status_code=402,
            content={
                "detail": f"This feature requires the {required.value.title()} plan.",
                "required_tier": required.value,
                "current_tier": current_tier.value,
            },
        )


__all__ = ["add_tier_gate_middleware"]
