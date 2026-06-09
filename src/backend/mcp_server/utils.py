"""Utility helpers for the Agentic Search MCP server."""

from __future__ import annotations

import logging
import os

import httpx
from fastmcp.server.auth.auth import AccessToken
from fastmcp.server.dependencies import get_access_token
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_http_client: httpx.AsyncClient | None = None


def build_web_base_url() -> str:
    """Return the base URL for the web backend."""
    override = os.getenv("API_SERVER_URL_OVERRIDE_FOR_HTTP_REQUESTS")
    if override:
        return override.rstrip("/")
    protocol = os.getenv("API_SERVER_PROTOCOL", "http")
    host = os.getenv("API_SERVER_HOST", "127.0.0.1")
    port = os.getenv("AGENTIC_SEARCH_WEB_PORT", "7860")
    return f"{protocol}://{host}:{port}"


def require_access_token() -> AccessToken:
    """Get and validate the access token from the current request."""
    access_token = get_access_token()
    if not access_token:
        raise ValueError(
            "MCP Server requires an access token to authenticate your request"
        )
    return access_token


def get_http_client() -> httpx.AsyncClient:
    """Return a shared async HTTP client."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=60.0)
    return _http_client


async def shutdown_http_client() -> None:
    """Close the shared HTTP client when the server shuts down."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


class DocumentSetEntry(BaseModel):
    """Minimal document-set shape surfaced to MCP clients."""

    name: str
    description: str | None = None
