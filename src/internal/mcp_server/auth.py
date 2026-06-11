"""Authentication helpers for the Agentic Search MCP server."""

import logging
from typing import Optional

from fastmcp.server.auth.auth import AccessToken
from fastmcp.server.auth.auth import TokenVerifier

from .utils import build_web_base_url
from .utils import get_http_client

logger = logging.getLogger(__name__)


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

        return AccessToken(
            token=token,
            client_id="mcp",
            scopes=["mcp:use"],
            expires_at=None,
            resource=None,
            claims={},
        )
