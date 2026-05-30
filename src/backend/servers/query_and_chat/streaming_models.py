"""SSE streaming packet types for the search and chat APIs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from .models import SearchDocWithContent


# ---------------------------------------------------------------------------
# Citation types (used by citation_processor and chat pipeline)
# ---------------------------------------------------------------------------


class CitationInfo(BaseModel):
    """Metadata for a single cited document emitted during streaming."""

    citation_number: int
    document_id: str


class GeneratedImage(BaseModel):
    """A single generated image returned by an image-generation tool."""

    url: str
    alt_text: str | None = None


# ---------------------------------------------------------------------------
# Chat message ID types
# ---------------------------------------------------------------------------


class MessageResponseIDInfo(BaseModel):
    """IDs assigned to the assistant message after it is persisted."""

    user_message_id: int | None
    reserved_assistant_message_id: int


class MultiModelMessageResponseIDInfo(BaseModel):
    """IDs for multi-model responses (one per model)."""

    user_message_id: int | None
    reserved_assistant_message_ids: list[int]


# ---------------------------------------------------------------------------
# Generic streaming packet wrapper
# ---------------------------------------------------------------------------


class Packet(BaseModel):
    """Generic streaming packet that wraps any serialisable payload."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    placement: Any | None = None
    obj: Any


# ---------------------------------------------------------------------------
# Control / debug packets
# ---------------------------------------------------------------------------


class OverallStop(BaseModel):
    type: Literal["stop"] = "stop"


class TopLevelBranching(BaseModel):
    """Signals the start of parallel tool-call branches."""

    num_parallel_branches: int


class ToolCallArgumentDelta(BaseModel):
    """Incremental token of a tool-call argument during streaming."""

    tool_call_id: str
    delta: str


class ToolCallDebug(BaseModel):
    """Debug payload emitted in integration-test mode for each tool call."""

    tool_call_id: str
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Search result streaming packets
# ---------------------------------------------------------------------------


class SearchQueriesPacket(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["search_queries"] = "search_queries"
    all_executed_queries: list[str]


class SearchDocsPacket(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["search_docs"] = "search_docs"
    search_docs: list[SearchDocWithContent]


class SearchErrorPacket(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["search_error"] = "search_error"
    error: str


class LLMSelectedDocsPacket(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["llm_selected_docs"] = "llm_selected_docs"
    llm_selected_doc_ids: list[str] | None


__all__ = [
    "CitationInfo",
    "GeneratedImage",
    "LLMSelectedDocsPacket",
    "MessageResponseIDInfo",
    "MultiModelMessageResponseIDInfo",
    "OverallStop",
    "Packet",
    "SearchDocsPacket",
    "SearchErrorPacket",
    "SearchQueriesPacket",
    "ToolCallArgumentDelta",
    "ToolCallDebug",
    "TopLevelBranching",
]
