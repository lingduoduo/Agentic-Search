"""OAuth admin API.

py.
Changes:
- Redis session store replaced with an in-memory dict (TTL enforced on access).
- Connector credential storage callbacks are stubbed with HTTP 501 — this
  deployment has no connector credential DB.
- require_permission / User replaced with the project's _require_admin pattern.
"""

from __future__ import annotations

import base64
import logging
import time
import threading
import uuid

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from src.backend.auth import AuthenticatedUser
from src.backend.configs import AppSettings
from src.backend.servers.oauth.confluence_cloud import ConfluenceCloudOAuth
from src.backend.servers.oauth.google_drive import GoogleDriveOAuth
from src.backend.servers.oauth.slack import SlackOAuth
from src.backend.servers._auth import make_require_admin

logger = logging.getLogger(__name__)

_OAUTH_SESSION_TTL = 600  # seconds

_session_lock = threading.Lock()
_sessions: dict[str, tuple[str, float]] = {}  # uuid_str -> (json, expiry)

_SUPPORTED_CONNECTORS = {"slack", "confluence", "google_drive"}


def _store_session(uuid_str: str, session_json: str) -> None:
    now = time.monotonic()
    with _session_lock:
        # Purge expired entries to prevent unbounded growth
        expired = [k for k, (_, exp) in _sessions.items() if exp <= now]
        for k in expired:
            del _sessions[k]
        _sessions[uuid_str] = (session_json, now + _OAUTH_SESSION_TTL)


def _pop_session(uuid_str: str) -> str | None:
    with _session_lock:
        entry = _sessions.pop(uuid_str, None)
    if not entry:
        return None
    session_json, expiry = entry
    return session_json if time.monotonic() < expiry else None


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
        user: AuthenticatedUser = Depends(_require_admin),
    ) -> JSONResponse:
        """Generate the OAuth authorization URL for the given connector.

        The frontend redirects the user's browser to the returned URL.
        Session state is stored in memory for up to 10 minutes.
        """
        oauth_uuid = uuid.uuid4()
        oauth_uuid_str = str(oauth_uuid)
        oauth_state = (
            base64.urlsafe_b64encode(oauth_uuid.bytes).rstrip(b"=").decode("utf-8")
        )

        connector_lower = connector.lower()
        email = user.email or ""

        if connector_lower == "slack":
            if not SlackOAuth.CLIENT_ID:
                raise HTTPException(
                    status_code=500,
                    detail="Slack OAuth client ID is not configured (OAUTH_SLACK_CLIENT_ID).",
                )
            oauth_url = SlackOAuth.generate_oauth_url(oauth_state, web_domain, dev_mode)
            session_json = SlackOAuth.session_dump_json(email, redirect_on_success)

        elif connector_lower == "confluence":
            if not ConfluenceCloudOAuth.CLIENT_ID:
                raise HTTPException(
                    status_code=500,
                    detail="Confluence OAuth client ID is not configured (OAUTH_CONFLUENCE_CLOUD_CLIENT_ID).",
                )
            oauth_url = ConfluenceCloudOAuth.generate_oauth_url(
                oauth_state, web_domain, dev_mode
            )
            session_json = ConfluenceCloudOAuth.session_dump_json(
                email, redirect_on_success
            )

        elif connector_lower == "google_drive":
            if not GoogleDriveOAuth.CLIENT_ID:
                raise HTTPException(
                    status_code=500,
                    detail="Google Drive OAuth client ID is not configured (OAUTH_GOOGLE_DRIVE_CLIENT_ID).",
                )
            oauth_url = GoogleDriveOAuth.generate_oauth_url(
                oauth_state, web_domain, dev_mode
            )
            session_json = GoogleDriveOAuth.session_dump_json(
                email, redirect_on_success
            )

        else:
            raise HTTPException(
                status_code=404,
                detail=f"Connector '{connector}' does not have OAuth implemented. "
                f"Supported: {sorted(_SUPPORTED_CONNECTORS)}",
            )

        _store_session(oauth_uuid_str, session_json)
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
