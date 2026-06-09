"""Resource exposing document sets — returns empty list (not implemented in this repo)."""

from __future__ import annotations

import json
import logging

from ..api import mcp_server

logger = logging.getLogger(__name__)


@mcp_server.resource(
    "resource://document_sets",
    name="document_sets",
    description=(
        "Enumerate document sets available for scoping searches. "
        "Document sets are not yet configured in this deployment — returns an empty list."
    ),
    mime_type="application/json",
)
async def document_sets_resource() -> str:
    """Return an empty list — document sets are not used in this repo."""
    logger.debug("MCP Server: document_sets resource: no sets configured")
    return json.dumps([])
