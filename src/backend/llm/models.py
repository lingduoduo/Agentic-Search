"""LLM message and response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TextContentPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageUrlDetail(str):
    pass


class ImageContentPart(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: dict[str, str]


ContentPart = TextContentPart | ImageContentPart


class FunctionCall(BaseModel):
    name: str
    arguments: str = ""


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[ContentPart]


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str


ChatCompletionMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage


class ReasoningEffort(str):
    AUTO = "auto"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
