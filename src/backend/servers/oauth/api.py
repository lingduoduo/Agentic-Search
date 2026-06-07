"""OAuth admin API.

py.
Changes:
- Connector credential storage callbacks return HTTP 501 because this deployment
  has no connector credential DB.
- require_permission / User replaced with the project's _require_admin pattern.
"""

from __future__ import annotations

import base64
import logging
import uuid

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from src.backend.auth import AuthenticatedUser
from src.backend.configs import AppSettings
from src.backend.servers.oauth import OAUTH_PROVIDERS, get_oauth_provider
from src.backend.servers._auth import make_require_admin

logger = logging.getLogger(__name__)


def create_oauth_router(app_settings: AppSettings) -> APIRouter:
    """Return an APIRouter for OAuth connector authorization flows."""

    router = APIRouter(prefix="/oauth", tags=["oauth"])

    _require_admin = make_require_admin(app_settings)

    dev_mode = getattr(app_settings, "dev_mode", False)
    web_domain = app_settings.web_domain

    @router.post("/prepare-authorization-request")
    def prepare_authorization_request(
        connector: str,
        redirect_on_success: str | None = None,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> JSONResponse:
        """Generate the OAuth authorization URL for the given connector.

        The frontend redirects the user's browser to the returned URL.
        Callback persistence is unavailable in this deployment, so no server-side
        session state is retained.
        """
        oauth_uuid = uuid.uuid4()
        oauth_state = (
            base64.urlsafe_b64encode(oauth_uuid.bytes).rstrip(b"=").decode("utf-8")
        )

        provider = get_oauth_provider(connector)
        if provider is None:
            raise HTTPException(
                status_code=404,
                detail=f"Connector '{connector}' does not have OAuth implemented. "
                f"Supported: {sorted(OAUTH_PROVIDERS)}",
            )
        if not provider.CLIENT_ID:
            raise HTTPException(
                status_code=500,
                detail=f"{provider.DISPLAY_NAME} OAuth client ID is not configured "
                f"({provider.CLIENT_ID_ENV_VAR}).",
            )

        oauth_url = provider.generate_oauth_url(oauth_state, web_domain, dev_mode)
        del redirect_on_success
        return JSONResponse(content={"url": oauth_url})

    @router.post("/connector/{connector}/callback")
    def oauth_callback(
        connector: str,
        code: str,
        state: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> JSONResponse:
        """OAuth callback handler.

        Credential persistence requires a connector credential database which
        is not available in this deployment. Returns HTTP 501.
        """
        raise HTTPException(
            status_code=501,
            detail=(
                f"OAuth callback for '{connector}' is not supported in this deployment. "
                "Connector credential storage requires additional infrastructure."
            ),
        )

    return router


__all__ = ["create_oauth_router"]
