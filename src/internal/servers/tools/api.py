"""Tool management REST API.

Endpoints (prefix /admin/tools):

  GET    /                      — list all registered tools
  POST   /openapi               — register tools from an OpenAPI JSON body or URL
  DELETE /openapi/{provider_id} — remove all tools from an OpenAPI provider
  GET    /{name}                — get one tool's full schema
  POST   /{name}/invoke         — test-invoke a tool with supplied arguments
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field

from src.internal.auth import AuthenticatedUser
from src.internal.configs import AppSettings
from src.internal.servers._auth import make_require_admin
from src.internal.tools.registry import tool_registry

logger = logging.getLogger(__name__)


def _try_sync_to_mcp(name: str) -> None:
    """Optionally sync a newly registered tool to the running MCP server."""
    try:
        from src.internal.mcp_server.tools.dynamic import sync_tool_to_mcp

        sync_tool_to_mcp(name)
    except Exception as exc:
        logger.debug("Could not sync tool %r to MCP: %s", name, exc)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ToolView(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    source: str
    provider_id: str | None


class OpenAPIRegisterRequest(BaseModel):
    name: str = Field(..., description="Human-readable name for this API provider")
    openapi_json: str = Field(..., description="OpenAPI 3.x JSON schema as a string")
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="HTTP headers to include in every API call (e.g. Authorization)",
    )


class OpenAPIRegisterResponse(BaseModel):
    provider_id: str
    tool_names: list[str]


class InvokeRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


class InvokeResponse(BaseModel):
    response: str
    raw: Any
    errors: list[str]


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_tools_router(settings: AppSettings) -> APIRouter:
    router = APIRouter(prefix="/admin/tools", tags=["tools"])
    require_admin = make_require_admin(settings)

    @router.get("", response_model=list[ToolView])
    async def list_tools(
        _: AuthenticatedUser = Depends(require_admin),
    ) -> list[ToolView]:
        return [ToolView(**s) for s in tool_registry.all_summaries()]

    # Must be declared before /{name} so FastAPI doesn't match "openapi" as a name
    @router.post(
        "/openapi",
        response_model=OpenAPIRegisterResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def register_openapi(
        req: OpenAPIRegisterRequest,
        _: AuthenticatedUser = Depends(require_admin),
    ) -> OpenAPIRegisterResponse:
        try:
            tool_names = tool_registry.register_from_openapi(
                req.openapi_json,
                name=req.name,
                headers=req.headers or None,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )

        # Find the provider_id by looking up the first registered tool
        provider_id = ""
        if tool_names:
            entry = tool_registry._entries.get(tool_names[0])
            if entry:
                provider_id = entry.provider_id or ""

        for name in tool_names:
            _try_sync_to_mcp(name)

        return OpenAPIRegisterResponse(provider_id=provider_id, tool_names=tool_names)

    @router.delete(
        "/openapi/{provider_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        response_class=Response,
        response_model=None,
    )
    async def delete_openapi_provider(
        provider_id: str,
        _: AuthenticatedUser = Depends(require_admin),
    ) -> None:
        removed = tool_registry.unregister_provider(provider_id)
        if not removed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider {provider_id!r} not found.",
            )

    @router.get("/{name}", response_model=ToolView)
    async def get_tool(
        name: str,
        _: AuthenticatedUser = Depends(require_admin),
    ) -> ToolView:
        summary = tool_registry.tool_summary(name)
        if summary is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool {name!r} not found.",
            )
        return ToolView(**summary)

    @router.post("/{name}/invoke", response_model=InvokeResponse)
    async def invoke_tool(
        name: str,
        req: InvokeRequest,
        _: AuthenticatedUser = Depends(require_admin),
    ) -> InvokeResponse:
        if tool_registry.get(name) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool {name!r} not found.",
            )
        response_text, raw, errors = await tool_registry.invoke(name, req.arguments)
        return InvokeResponse(response=response_text, raw=raw, errors=errors)

    return router
