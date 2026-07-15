"""Concrete LLM implementation backed by any OpenAI-compatible HTTP API.

Any provider that exposes the OpenAI streaming chat-completions protocol
(OpenAI, Azure OpenAI, Anthropic via compatibility layer, Ollama, LiteLLM
proxy, etc.) can be used by setting the GEN_AI_* environment variables.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Iterator
from time import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter

from src.context.models import LLMResponse
from src.context.structured_output import (
    SchemaUnsupportedError,
    StructuredCompletionMetadata,
    StructuredOutputCapability,
    StructuredOutputRequest,
)

from .interfaces import LLM, LLMConfig, LLMUserIdentity, ToolChoiceOptions
from .model_response import (
    ChatCompletionDeltaToolCall,
    Delta,
    FunctionCall,
    ModelResponseStream,
    StreamingChoice,
    Usage,
)
from .models import LanguageModelInput, ReasoningEffort

logger = logging.getLogger(__name__)

_TOOL_CHOICE_MAP: dict[ToolChoiceOptions, str] = {
    ToolChoiceOptions.AUTO: "auto",
    ToolChoiceOptions.NONE: "none",
    ToolChoiceOptions.REQUIRED: "required",
}


def _build_tool_call_delta(raw: dict) -> ChatCompletionDeltaToolCall:
    fn = raw.get("function") or {}
    return ChatCompletionDeltaToolCall(
        index=raw.get("index", 0),
        id=raw.get("id"),
        type=raw.get("type", "function"),
        function=FunctionCall(
            name=fn.get("name") or "",
            arguments=fn.get("arguments") or "",
        ),
    )


def _parse_sse_chunk(line: str, chunk_id: str = "") -> ModelResponseStream | None:
    """Parse one SSE data line into a ModelResponseStream; return None for non-data lines."""
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
    delta = Delta(
        content=delta_raw.get("content"),
        reasoning_content=delta_raw.get("reasoning_content"),
        tool_calls=tool_calls,
    )
    choice = StreamingChoice(
        finish_reason=choice_raw.get("finish_reason"),
        delta=delta,
    )

    usage_raw = data.get("usage")
    usage: Usage | None = None
    if usage_raw:
        usage = Usage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
            cache_read_input_tokens=usage_raw.get("cache_read_input_tokens", 0),
            cache_creation_input_tokens=usage_raw.get("cache_creation_input_tokens", 0),
        )

    return ModelResponseStream(
        id=chunk_id,
        created=str(int(time())),
        choice=choice,
        usage=usage,
    )


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

    @property
    def structured_output_capability(self) -> StructuredOutputCapability:
        if self._config.model_provider == "openai":
            return StructuredOutputCapability.JSON_SCHEMA
        configured = (self._config.custom_config or {}).get("supports_json_schema")
        if (
            self._config.model_provider in {"openai_compatible", "openai-compatible"}
            and configured is not None
            and configured.lower() == "true"
        ):
            return StructuredOutputCapability.JSON_SCHEMA
        return StructuredOutputCapability.PROMPT_ONLY

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
    ) -> Iterator[ModelResponseStream]:
        messages = self._normalise_messages(prompt)
        body: dict[str, Any] = {
            "model": self._config.model_name,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens:
            body["max_tokens"] = max_tokens
        if tools:
            body["tools"] = tools
            tc = tool_choice or ToolChoiceOptions.AUTO
            body["tool_choice"] = _TOOL_CHOICE_MAP.get(tc, "auto")

        timeout = timeout_override or 120
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
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
                chunk = _parse_sse_chunk(raw_line, chunk_id)
                if chunk is not None:
                    yield chunk
        finally:
            resp.close()

    @staticmethod
    def _normalise_messages(prompt: LanguageModelInput) -> list[dict]:
        """Convert the various prompt shapes into messages[]."""
        if isinstance(prompt, list):
            out = []
            for m in prompt:
                if isinstance(m, dict):
                    out.append(m)
                elif hasattr(m, "role") and hasattr(m, "content"):
                    entry: dict[str, Any] = {
                        "role": m.role if isinstance(m.role, str) else m.role.value,
                        "content": (
                            m.content
                            if not isinstance(m.content, list)
                            else [
                                part.model_dump(exclude_none=True) for part in m.content
                            ]
                        ),
                    }
                    if getattr(m, "tool_calls", None):
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
                    if getattr(m, "tool_call_id", None):
                        entry["tool_call_id"] = m.tool_call_id
                    out.append(entry)
                else:
                    out.append({"role": "user", "content": str(m)})
            return out
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if hasattr(prompt, "role"):
            return [{"role": prompt.role, "content": prompt.content}]
        return [{"role": "user", "content": str(prompt)}]

    def complete(
        self, messages: LanguageModelInput, **kwargs: Any
    ) -> LLMResponse | str:
        """Non-streaming completion — used for short utility calls."""
        # Deferred import: src.internal.servers.web's package __init__ imports
        # this module (via app.py), so a top-level import here would be circular.
        from src.internal.servers.web import request_capture as _capture

        normalised = self._normalise_messages(messages)
        body: dict[str, Any] = {
            "model": self._config.model_name,
            "messages": normalised,
            "stream": False,
        }
        max_tokens = kwargs.get("max_tokens")
        if max_tokens:
            body["max_tokens"] = max_tokens
        temperature = kwargs.get("temperature")
        if temperature is not None:
            body["temperature"] = temperature
        structured_output: StructuredOutputRequest | None = kwargs.get(
            "structured_output"
        )
        schema_applied = bool(
            structured_output
            and self.structured_output_capability
            is StructuredOutputCapability.JSON_SCHEMA
        )
        if structured_output and schema_applied:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": structured_output.name,
                    "strict": structured_output.strict,
                    "schema": structured_output.schema,
                },
            }
        timeout = kwargs.get("timeout_override") or 30
        try:
            resp = self._session.post(
                self._endpoint,
                headers=self._headers,
                json=body,
                timeout=timeout,
            )
            resp.raise_for_status()
        except requests.HTTPError as exc:
            response = exc.response
            provider_error = response.text.lower() if response is not None else ""
            schema_parameter_unsupported = bool(
                re.search(
                    r"(?:unknown|unsupported)\s+"
                    r"(?:(?:parameter|field)\s*)?[:=]?\s*[\"'`]?"
                    r"(?:response_format|json_schema)\b"
                    r"|\b(?:response_format|json_schema)\b"
                    r"(?:\s+\w+){0,3}\s+(?:unknown|unsupported)\b",
                    provider_error,
                )
            )
            if (
                schema_applied
                and response is not None
                and response.status_code == 400
                and schema_parameter_unsupported
            ):
                raise SchemaUnsupportedError(
                    "Provider does not support JSON Schema response formatting"
                ) from None
            raise
        data = resp.json()
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        refusal = message.get("refusal")
        finish_reason = choice.get("finish_reason")
        metadata = StructuredCompletionMetadata(
            requested=structured_output is not None,
            applied=schema_applied,
            refused=bool(refusal),
            incomplete_reason="length" if finish_reason == "length" else None,
        )
        capture_payload: dict[str, Any]
        if structured_output is not None:
            capture_payload = {
                "model": self._config.model_name,
                "structured": {
                    "requested": metadata.requested,
                    "applied": metadata.applied,
                    "downgraded": metadata.downgraded,
                    "refused": metadata.refused,
                    "incomplete_reason": metadata.incomplete_reason,
                },
            }
        else:
            capture_payload = {
                "model": self._config.model_name,
                "messages": normalised,
                "completion": content,
                "usage": data.get("usage"),
            }
        _capture.record_stage(
            "llm",
            "complete",
            capture_payload,
        )
        if structured_output is not None:
            return LLMResponse(
                text=content,
                raw={"refusal": refusal, "finish_reason": finish_reason},
                structured=metadata,
            )
        return content
