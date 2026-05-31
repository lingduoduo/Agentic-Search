"""Tests for OpenAICompatibleLLM and get_llm_for_persona."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


from src.backend.llm.interfaces import LLMConfig, ToolChoiceOptions
from src.backend.llm.providers import (
    OpenAICompatibleLLM,
    _parse_sse_chunk,
)


# ---------------------------------------------------------------------------
# _parse_sse_chunk
# ---------------------------------------------------------------------------


def test_parse_sse_chunk_content():
    line = 'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}'
    chunk = _parse_sse_chunk(line)
    assert chunk is not None
    assert chunk.choice.delta.content == "hello"
    assert chunk.choice.finish_reason is None
    assert chunk.usage is None


def test_parse_sse_chunk_done():
    assert _parse_sse_chunk("data: [DONE]") is None


def test_parse_sse_chunk_non_data():
    assert _parse_sse_chunk(": keep-alive") is None


def test_parse_sse_chunk_finish_reason():
    line = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
    chunk = _parse_sse_chunk(line)
    assert chunk is not None
    assert chunk.choice.finish_reason == "stop"


def test_parse_sse_chunk_usage():
    line = json.dumps(
        {
            "choices": [{"delta": {"content": "hi"}, "finish_reason": None}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        }
    )
    chunk = _parse_sse_chunk(f"data: {line}")
    assert chunk is not None
    assert chunk.usage is not None
    assert chunk.usage.prompt_tokens == 10
    assert chunk.usage.completion_tokens == 5


def test_parse_sse_chunk_tool_call():
    line = json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"q":'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
    )
    chunk = _parse_sse_chunk(f"data: {line}")
    assert chunk is not None
    tc = chunk.choice.delta.tool_calls[0]
    assert tc.function_name == "search"
    assert tc.id == "call_abc"


# ---------------------------------------------------------------------------
# OpenAICompatibleLLM.stream
# ---------------------------------------------------------------------------


def _make_config(**kwargs):
    return LLMConfig(
        model_provider="openai",
        model_name="gpt-4o-mini",
        api_key="test-key",
        **kwargs,
    )


def _mock_response(*data_lines: str):
    """Build a mock requests.Response that streams the given SSE lines."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.iter_lines.return_value = iter(data_lines)
    return resp


def test_stream_yields_content_chunks():
    config = _make_config()
    llm = OpenAICompatibleLLM(config)
    lines = [
        'data: {"choices":[{"delta":{"content":"hello "},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":"world"},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    with patch("requests.Session.post", return_value=_mock_response(*lines)):
        chunks = list(llm.stream(prompt="hi"))
    assert len(chunks) == 2
    assert chunks[0].choice.delta.content == "hello "
    assert chunks[1].choice.finish_reason == "stop"


def test_stream_normalises_string_prompt():
    config = _make_config()
    llm = OpenAICompatibleLLM(config)
    with patch(
        "requests.Session.post", return_value=_mock_response("data: [DONE]")
    ) as mock_post:
        list(llm.stream(prompt="hello"))
    body = mock_post.call_args.kwargs["json"]
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_stream_normalises_dict_messages():
    config = _make_config()
    llm = OpenAICompatibleLLM(config)
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ]
    with patch(
        "requests.Session.post", return_value=_mock_response("data: [DONE]")
    ) as mock_post:
        list(llm.stream(prompt=msgs))
    body = mock_post.call_args.kwargs["json"]
    assert body["messages"] == msgs


def test_stream_includes_tools():
    config = _make_config()
    llm = OpenAICompatibleLLM(config)
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    with patch(
        "requests.Session.post", return_value=_mock_response("data: [DONE]")
    ) as mock_post:
        list(llm.stream(prompt="hi", tools=tools, tool_choice=ToolChoiceOptions.AUTO))
    body = mock_post.call_args.kwargs["json"]
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"


def test_stream_no_tools_omits_tool_fields():
    config = _make_config()
    llm = OpenAICompatibleLLM(config)
    with patch(
        "requests.Session.post", return_value=_mock_response("data: [DONE]")
    ) as mock_post:
        list(llm.stream(prompt="hi"))
    body = mock_post.call_args.kwargs["json"]
    assert "tools" not in body
    assert "tool_choice" not in body


def test_stream_custom_base_url():
    config = LLMConfig(
        model_provider="ollama_chat",
        model_name="llama3",
        api_base="http://localhost:11434/v1",
    )
    llm = OpenAICompatibleLLM(config)
    assert llm._endpoint == "http://localhost:11434/v1/chat/completions"
    assert "Authorization" not in llm._headers


# ---------------------------------------------------------------------------
# get_llm_for_persona
# ---------------------------------------------------------------------------


def test_get_llm_for_persona_uses_env_defaults(monkeypatch):
    monkeypatch.setenv("GEN_AI_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("GEN_AI_MODEL_VERSION", "gpt-4o-mini")
    monkeypatch.setenv("GEN_AI_API_KEY", "sk-test")

    from src.backend.chat.process_message import get_llm_for_persona

    llm = get_llm_for_persona(persona=None, user=None)
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.config.model_provider == "openai"
    assert llm.config.model_name == "gpt-4o-mini"
    assert llm.config.api_key == "sk-test"


def test_get_llm_for_persona_override_respected(monkeypatch):
    monkeypatch.setenv("GEN_AI_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("GEN_AI_MODEL_VERSION", "gpt-4o-mini")
    monkeypatch.setenv("GEN_AI_API_KEY", "sk-test")

    from src.backend.chat.process_message import LLMOverride, get_llm_for_persona

    override = LLMOverride(
        model_provider="anthropic", model_version="claude-3-5-haiku-20241022"
    )
    llm = get_llm_for_persona(persona=None, user=None, llm_override=override)
    assert llm.config.model_provider == "anthropic"
    assert llm.config.model_name == "claude-3-5-haiku-20241022"


def test_get_llm_for_persona_partial_override(monkeypatch):
    monkeypatch.setenv("GEN_AI_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("GEN_AI_MODEL_VERSION", "gpt-4o-mini")
    monkeypatch.setenv("GEN_AI_API_KEY", "sk-test")

    from src.backend.chat.process_message import LLMOverride, get_llm_for_persona

    # Only override the model version, keep the provider from defaults
    override = LLMOverride(model_version="gpt-4o")
    llm = get_llm_for_persona(persona=None, user=None, llm_override=override)
    assert llm.config.model_provider == "openai"
    assert llm.config.model_name == "gpt-4o"
