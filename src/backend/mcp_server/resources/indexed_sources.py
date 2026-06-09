"""Resource exposing available document sources in Agentic Search."""

from __future__ import annotations

import json
import logging
import os

from ..api import mcp_server

logger = logging.getLogger(__name__)


@mcp_server.resource(
    "resource://indexed_sources",
    name="indexed_sources",
    description=(
        "Enumerate available document sources for the `search_indexed_documents` tool. "
        "Always includes 'retrieval' (local corpus). Web providers are listed when their "
        "API keys are configured."
    ),
    mime_type="application/json",
)
async def indexed_sources_resource() -> str:
    """Return the list of available source types for search filtering."""
    sources = ["retrieval"]
    if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID"):
        sources.append("google")
    if os.getenv("SERPAPI_API_KEY") or os.getenv("SERP_API_KEY"):
        sources.append("serpapi")
    if os.getenv("SERPER_API_KEY"):
        sources.append("serper")

    logger.info(
        "MCP Server: indexed_sources resource returning %d entries", len(sources)
    )
    return json.dumps(sorted(sources))
