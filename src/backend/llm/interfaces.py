"""Abstract LLM interface.

Defines the Protocol that all LLM backends must implement so that the chat
pipeline (llm_loop, llm_step, compression) stays provider-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolChoiceOptions(str, Enum):
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


@dataclass(frozen=True)
class LLMConfig:
    model_provider: str
    model_name: str
    max_input_tokens: int = 8192
    api_key: str | None = None
    api_base: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMUserIdentity:
    user_id: str | None = None
    email: str | None = None


class LLM(ABC):
    """Minimal abstract LLM interface used by the chat pipeline."""

    @property
    @abstractmethod
    def config(self) -> LLMConfig: ...

    @abstractmethod
    def stream(
        self,
        prompt: Any,
        *,
        tools: list[dict] | None = None,
        tool_choice: ToolChoiceOptions = ToolChoiceOptions.AUTO,
        **kwargs: Any,
    ) -> Iterator[Any]: ...


# Alias used in some callers
LanguageModelInput = Any


class LlmProviderNames:
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    GOOGLE = "google"
    COHERE = "cohere"


def is_true_openai_model(provider: str, model: str) -> bool:
    return provider == LlmProviderNames.OPENAI
