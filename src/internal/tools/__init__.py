"""Chat loop tool interface and built-in tool stubs."""

from .interface import ChatTool

# Backward-compat re-exports (canonical locations noted)
# Chat models live in src.internal.chat.tool_models
from src.internal.chat.tool_models import (
    ChatFile,
    SearchToolUsage,
    ToolCallInfo,
    ToolCallKickoff,
    ToolResponse,
)

# OpenAPI schema lives in src.tools.openapi_schema
from src.tools.openapi_schema import (
    OpenAPISchema,
    ParameterIn,
    ParameterType,
    ParameterTypeMap,
)

__all__ = [
    "ChatFile",
    "ChatTool",
    "OpenAPISchema",
    "ParameterIn",
    "ParameterType",
    "ParameterTypeMap",
    "SearchToolUsage",
    "ToolCallInfo",
    "ToolCallKickoff",
    "ToolResponse",
]
