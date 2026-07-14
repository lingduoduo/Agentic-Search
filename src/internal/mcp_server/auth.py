"""Authentication helpers for the Agentic Search MCP server."""

import logging
import time
from typing import Optional

from fastmcp.server.auth.auth import AccessToken
from fastmcp.server.auth.auth import TokenVerifier

from src.internal.auth import generate_user_jwt_token

from .utils import build_web_base_url
from .utils import get_http_client

logger = logging.getLogger(__name__)

_DOWNSTREAM_TOKEN_TTL_SECONDS = 300


class AgenticSearchTokenVerifier(TokenVerifier):
    """Validates bearer tokens by delegating to the web backend /me endpoint."""

    async def verify_token(self, token: str) -> Optional[AccessToken]:
        try:
            response = await get_http_client().get(
                f"{build_web_base_url()}/me",
                headers={"Authorization": f"Bearer {token}"},
            )
        except Exception as exc:
            logger.error(
                "MCP server failed to reach /me for authentication: %s",
                exc,
                exc_info=True,
            )
            return None

        if response.status_code != 200:
            logger.warning(
                "Web backend rejected MCP auth token with status %s",
                response.status_code,
            )
            return None

        try:
            identity = response.json()
        except (TypeError, ValueError):
            logger.warning("Web backend returned invalid MCP identity JSON")
            return None

        if not isinstance(identity, dict):
            logger.warning("Web backend returned a malformed MCP identity")
            return None
        user_id = identity.get("id")
        email = identity.get("email")
        role = identity.get("role")
        if not isinstance(user_id, str) or not user_id.strip():
            logger.warning("Web backend MCP identity is missing a valid user id")
            return None
        if email is not None and not isinstance(email, str):
            logger.warning("Web backend MCP identity has an invalid email")
            return None
        if role is not None and not isinstance(role, str):
            logger.warning("Web backend MCP identity has an invalid role")
            return None

        downstream_token = generate_user_jwt_token(
            user_id=user_id.strip(),
            email=email,
            expires_in_seconds=_DOWNSTREAM_TOKEN_TTL_SECONDS,
            extra={"role": role} if role else None,
        )

        return AccessToken(
            token=downstream_token,
            client_id="mcp",
            scopes=["mcp:use"],
            expires_at=int(time.time()) + _DOWNSTREAM_TOKEN_TTL_SECONDS,
            resource=None,
            claims={"sub": user_id.strip(), "email": email, "role": role},
        )
