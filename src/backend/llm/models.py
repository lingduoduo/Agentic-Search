"""LLM message and response models."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class ToolChoiceOptions(str, Enum):
    REQUIRED = "required"
    AUTO = "auto"
    NONE = "none"


class ReasoningEffort(str, Enum):
    """Reasoning effort levels for models that support extended thinking.

    Different providers map these values differently:
    - OpenAI: Uses "low", "medium", "high" directly for reasoning_effort
    - Claude: Uses budget_tokens with different values for each level
    - Gemini: Uses "none", "low", "medium", "high" for thinking_budget (via litellm mapping)
    """

    AUTO = "auto"
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# OpenAI reasoning effort mapping
# Note: OpenAI API does not support "auto" - valid values are: none, minimal, low, medium, high, xhigh
OPENAI_REASONING_EFFORT: dict[ReasoningEffort, str] = {
    ReasoningEffort.AUTO: "medium",
    ReasoningEffort.OFF: "none",
    ReasoningEffort.LOW: "low",
    ReasoningEffort.MEDIUM: "medium",
    ReasoningEffort.HIGH: "high",
}

# Anthropic reasoning effort to budget tokens mapping
ANTHROPIC_REASONING_EFFORT_BUDGET: dict[ReasoningEffort, int] = {
    ReasoningEffort.AUTO: 2048,
    ReasoningEffort.LOW: 1024,
    ReasoningEffort.MEDIUM: 2048,
    ReasoningEffort.HIGH: 4096,
}

# Newer Anthropic models (Claude Opus 4.7+) use adaptive thinking with
# output_config.effort instead of thinking.type.enabled + budget_tokens.
ANTHROPIC_ADAPTIVE_REASONING_EFFORT: dict[ReasoningEffort, str] = {
    ReasoningEffort.AUTO: "medium",
    ReasoningEffort.LOW: "low",
    ReasoningEffort.MEDIUM: "medium",
    ReasoningEffort.HIGH: "high",
}


class TextContentPart(BaseModel):
    type: Literal["text"] = "text"
    text: str
    # Some providers (e.g. Anthropic/Gemini) support prompt caching controls on content blocks.
    cache_control: dict | None = None


class ImageUrlDetail(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = None


class ImageContentPart(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrlDetail


ContentPart = TextContentPart | ImageContentPart


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    type: Literal["function"] = "function"
    id: str
    function: FunctionCall


# Base class for all messages — providers that support prompt caching
# accept cache_control at the message level (passed through via LiteLLM).
class CacheableMessage(BaseModel):
    cache_control: dict | None = None


class SystemMessage(CacheableMessage):
    role: Literal["system"] = "system"
    content: str


class UserMessage(CacheableMessage):
    role: Literal["user"] = "user"
    content: str | list[ContentPart]


class AssistantMessage(CacheableMessage):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolMessage(CacheableMessage):
    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str


ChatCompletionMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage

# Allows passing a list of messages or a single message.
LanguageModelInput = list[ChatCompletionMessage] | ChatCompletionMessage
