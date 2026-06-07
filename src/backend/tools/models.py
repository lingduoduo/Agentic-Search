"""Shared models used by the tool pipeline and chat loop."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.backend.document_index.models import SearchDoc

# Type alias for custom tool results (string, dict, or list).
ToolResultType = str | dict[str, Any] | list[Any]


class SearchToolUsage(BaseModel):
    """Describes how the search tool was invoked in this turn."""

    query: str
    num_results: int = 0


class ToolCallKickoff(BaseModel):
    """A pending tool call produced by the LLM before execution."""

    tool_call_id: str
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    placement: Any | None = None  # Placement from the server models

    def to_msg_str(self) -> str:
        import json

        return json.dumps(
            {"tool": self.tool_name, "args": self.tool_args}, separators=(",", ":")
        )


class ToolResponse(BaseModel):
    """Result of executing one tool call."""

    tool_call: ToolCallKickoff | None = None
    llm_facing_response: str = ""
    rich_response: Any | None = None


class ToolCallInfo(BaseModel):
    """Persisted record of one completed tool call for a chat message."""

    parent_tool_call_id: str | None = None
    turn_index: int = 0
    tab_index: int | None = None
    tool_name: str
    tool_call_id: str
    tool_id: int | None = None
    reasoning_tokens: str | None = None
    tool_call_arguments: dict[str, Any] = Field(default_factory=dict)
    tool_call_response: str = ""
    search_docs: list[SearchDoc] | None = None
    generated_images: list[Any] | None = None
    generated_files: list[Any] | None = None


class ChatFile(BaseModel):
    """A file attached to the current chat session."""

    file_id: str
    filename: str | None = None
    content_type: str = "text/plain"


class CustomToolCallSummary(BaseModel):
    tool_name: str
    result: str


class MemoryToolResponseSnapshot(BaseModel):
    memory_text: str
    operation: str = "add"
    memory_id: int | None = None
    index: int | None = None


class PythonToolRichResponse(BaseModel):
    output: str = ""
    generated_files: list[Any] | None = None


__all__ = [
    "ChatFile",
    "CustomToolCallSummary",
    "MemoryToolResponseSnapshot",
    "PythonToolRichResponse",
    "SearchToolUsage",
    "ToolCallInfo",
    "ToolCallKickoff",
    "ToolResponse",
    "ToolResultType",
]


class FinalImageGenerationResponse(BaseModel):
    generated_images: list[Any] = []


class MemoryToolResponse(BaseModel):
    memory_text: str
    index_to_replace: int | None = None
