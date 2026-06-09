"""MCP server with FastAPI wrapper."""

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from fastmcp import FastMCP
from starlette.datastructures import MutableHeaders
from starlette.middleware.base import RequestResponseEndpoint
from starlette.types import Receive
from starlette.types import Scope
from starlette.types import Send

from .auth import AgenticSearchTokenVerifier
from .utils import shutdown_http_client

logger = logging.getLogger(__name__)

mcp_server = FastMCP(
    name="Agentic Search MCP Server",
    version="1.0.0",
    auth=AgenticSearchTokenVerifier(),
)

# Import tools and resources AFTER mcp_server is created to avoid circular imports
from .resources import document_sets  # noqa: E402, F401
from .resources import indexed_sources  # noqa: E402, F401
from .tools import search  # noqa: E402, F401

logger.info("MCP server instance created")


def _get_cors_origins() -> list[str]:
    raw = os.getenv("MCP_SERVER_CORS_ORIGINS", "")
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_mcp_fastapi_app() -> FastAPI:
    """Create FastAPI app wrapping MCP server with auth and shared client lifecycle."""
    mcp_asgi_app = mcp_server.http_app(path="/")

    async def _ensure_streamable_accept_header(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Ensure Accept header includes types required by FastMCP streamable HTTP."""
        if scope.get("type") == "http":
            headers = MutableHeaders(scope=scope)
            accept = headers.get("accept", "")
            accept_lower = accept.lower()

            if (
                not accept
                or accept == "*/*"
                or "application/json" not in accept_lower
                or "text/event-stream" not in accept_lower
            ):
                headers["accept"] = "application/json, text/event-stream"

        await mcp_asgi_app(scope, receive, send)

    @asynccontextmanager
    async def combined_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        logger.info("MCP server starting up")
        try:
            async with mcp_asgi_app.lifespan(app):
                yield
        finally:
            logger.info("MCP server shutting down")
            await shutdown_http_client()

    app = FastAPI(
        title="Agentic Search MCP Server",
        description="HTTP POST transport with bearer auth delegated to /me",
        version="1.0.0",
        lifespan=combined_lifespan,
    )

    @app.middleware("http")
    async def health_check(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path.rstrip("/") == "/health":
            return JSONResponse({"status": "healthy", "service": "mcp_server"})
        return await call_next(request)

    cors_origins = _get_cors_origins()
    if cors_origins:
        logger.info("CORS origins: %s", cors_origins)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.mount("/", _ensure_streamable_accept_header)

    return app


mcp_app = create_mcp_fastapi_app()
