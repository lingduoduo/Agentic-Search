"""Abstract LLM interface.

Defines the abstract base class that all LLM backends must implement so that
the chat pipeline stays provider-agnostic.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .constants import LlmProviderNames
from .models import LanguageModelInput, ReasoningEffort, ToolChoiceOptions

if TYPE_CHECKING:
    from .model_response import ModelResponse


class LLMUserIdentity(BaseModel):
    user_id: str | None = None
    session_id: str | None = None


class LLMConfig(BaseModel):
    model_provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    deployment_name: str | None = None
    custom_config: dict[str, str] | None = None
    max_input_tokens: int = 8192
    # Disable Pydantic's protected "model_" namespace guard so fields like
    # model_provider / model_name are accepted without warnings.
    model_config = {"protected_namespaces": ()}


class LLM(abc.ABC):
    """Abstract base for every LLM backend used by the chat pipeline."""

    @property
    @abc.abstractmethod
    def config(self) -> LLMConfig: ...

    def invoke(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None = None,
        tool_choice: ToolChoiceOptions | None = None,
        structured_response_format: dict | None = None,
        timeout_override: int | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
        user_identity: LLMUserIdentity | None = None,
    ) -> "ModelResponse":
        raise NotImplementedError

    def stream(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None = None,
        tool_choice: ToolChoiceOptions | None = None,
        structured_response_format: dict | None = None,
        timeout_override: int | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
        user_identity: LLMUserIdentity | None = None,
    ) -> Iterator[Any]:
        raise NotImplementedError


def is_true_openai_model(provider: str, model: str) -> bool:
    """Returns True only for native OpenAI (not OpenAI-compatible proxies)."""
    return str(provider) == str(LlmProviderNames.OPENAI)
