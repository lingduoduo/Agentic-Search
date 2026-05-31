"""Concrete LLM implementations backed by OpenAI-compatible HTTP APIs.

Any provider that exposes the OpenAI streaming chat-completions protocol
(OpenAI, Azure OpenAI, Anthropic via compatibility layer, Ollama, LiteLLM
proxy, etc.) can be used by setting the GEN_AI_* environment variables.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import requests
from requests.adapters import HTTPAdapter

from .interfaces import LLM, LLMConfig, ToolChoiceOptions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stream chunk data classes
# These mirror the OpenAI streaming response shape that llm_step.py expects.
# ---------------------------------------------------------------------------


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class ToolCallDelta:
    index: int = 0
    id: str | None = None
    type: str | None = None
    function_name: str | None = None
    function_arguments: str = ""


@dataclass
class LLMDelta:
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCallDelta] = field(default_factory=list)


@dataclass
class LLMChoice:
    finish_reason: str | None = None
    delta: LLMDelta = field(default_factory=LLMDelta)


@dataclass
class LLMChunk:
    """Single SSE chunk yielded by LLM.stream()."""

    choice: LLMChoice = field(default_factory=LLMChoice)
    usage: LLMUsage | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOOL_CHOICE_MAP: dict[ToolChoiceOptions, str | dict] = {
    ToolChoiceOptions.AUTO: "auto",
    ToolChoiceOptions.NONE: "none",
    ToolChoiceOptions.REQUIRED: "required",
}


def _build_tool_call_delta(raw: dict) -> ToolCallDelta:
    fn = raw.get("function") or {}
    return ToolCallDelta(
        index=raw.get("index", 0),
        id=raw.get("id"),
        type=raw.get("type"),
        function_name=fn.get("name"),
        function_arguments=fn.get("arguments", ""),
    )


def _parse_sse_chunk(line: str) -> LLMChunk | None:
    """Parse one SSE data line into an LLMChunk; return None for non-data lines."""
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        logger.debug("Skipping non-JSON SSE line: %s", line[:120])
        return None

    choices = data.get("choices") or []
    choice_raw = choices[0] if choices else {}
    delta_raw = choice_raw.get("delta") or {}

    tool_calls = [
        _build_tool_call_delta(tc) for tc in (delta_raw.get("tool_calls") or [])
    ]
    delta = LLMDelta(
        content=delta_raw.get("content"),
        reasoning_content=delta_raw.get("reasoning_content"),
        tool_calls=tool_calls,
    )
    choice = LLMChoice(
        finish_reason=choice_raw.get("finish_reason"),
        delta=delta,
    )

    usage_raw = data.get("usage")
    usage: LLMUsage | None = None
    if usage_raw:
        usage = LLMUsage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            cache_read_input_tokens=usage_raw.get("cache_read_input_tokens", 0),
            cache_creation_input_tokens=usage_raw.get("cache_creation_input_tokens", 0),
        )

    return LLMChunk(choice=choice, usage=usage)


# ---------------------------------------------------------------------------
# Concrete provider
# ---------------------------------------------------------------------------


class OpenAICompatibleLLM(LLM):
    """LLM implementation that streams from any OpenAI-compatible endpoint.

    Supports OpenAI, Azure OpenAI, Anthropic (via openai-compat proxy),
    Ollama (/v1/chat/completions), LiteLLM proxy, and similar APIs.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        base = (config.api_base or "https://api.openai.com/v1").rstrip("/")
        self._endpoint = f"{base}/chat/completions"
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if config.api_key:
            self._headers["Authorization"] = f"Bearer {config.api_key}"
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=16)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def close(self) -> None:
        self._session.close()

    @property
    def config(self) -> LLMConfig:
        return self._config

    def stream(
        self,
        prompt: Any,
        *,
        tools: list[dict] | None = None,
        tool_choice: ToolChoiceOptions = ToolChoiceOptions.AUTO,
        **kwargs: Any,
    ) -> Iterator[LLMChunk]:
        messages = self._normalise_messages(prompt)
        body: dict[str, Any] = {
            "model": self._config.model_name,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        max_tokens = kwargs.get("max_tokens")
        if max_tokens:
            body["max_tokens"] = max_tokens
        if tools:
            body["tools"] = tools
            body["tool_choice"] = _TOOL_CHOICE_MAP.get(tool_choice, "auto")

        timeout = kwargs.get("timeout_override") or 120
        resp: requests.Response | None = None
        try:
            resp = self._session.post(
                self._endpoint,
                headers=self._headers,
                json=body,
                stream=True,
                timeout=timeout,
            )
            resp.raise_for_status()
        except requests.HTTPError as exc:
            logger.error(
                "LLM HTTP error %s from %s: %s",
                exc.response.status_code if exc.response else "?",
                self._endpoint,
                exc.response.text[:500] if exc.response else str(exc),
            )
            raise

        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                chunk = _parse_sse_chunk(raw_line)
                if chunk is not None:
                    yield chunk
        finally:
            resp.close()

    @staticmethod
    def _normalise_messages(prompt: Any) -> list[dict]:
        """Convert the various prompt shapes llm_step passes into messages[]."""
        if isinstance(prompt, list):
            out = []
            for m in prompt:
                if isinstance(m, dict):
                    out.append(m)
                elif hasattr(m, "role") and hasattr(m, "content"):
                    # LLM message dataclass / pydantic model
                    entry: dict[str, Any] = {
                        "role": m.role if isinstance(m.role, str) else m.role.value,
                        "content": m.content,
                    }
                    if hasattr(m, "tool_calls") and m.tool_calls:
                        entry["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in m.tool_calls
                        ]
                    if hasattr(m, "tool_call_id") and m.tool_call_id:
                        entry["tool_call_id"] = m.tool_call_id
                    out.append(entry)
                else:
                    out.append({"role": "user", "content": str(m)})
            return out
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return [{"role": "user", "content": str(prompt)}]
