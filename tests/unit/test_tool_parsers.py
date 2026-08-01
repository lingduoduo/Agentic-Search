"""Tool-call parsers decode model output before extracting calls."""

from __future__ import annotations

import pytest

from src.internal.tools.parsers import ToolParser


class _RecordingTokenizer:
    """Decodes to a fixed string and records how ``decode`` was called."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def decode(self, ids, **kwargs):
        self.calls.append(kwargs)
        return self.text


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["hermes", "json"])
async def test_parsers_strip_special_tokens_from_content(fmt):
    # Without skip_special_tokens the EOS marker lands in the user-visible answer.
    tokenizer = _RecordingTokenizer("FAISS is a similarity search library.")
    parser = ToolParser.get_tool_parser(fmt, tokenizer)

    content, calls = await parser.extract_tool_calls([1, 2, 3])

    assert calls == []
    assert content == "FAISS is a similarity search library."
    assert tokenizer.calls == [{"skip_special_tokens": True}]


@pytest.mark.asyncio
async def test_llama3_parser_keeps_special_tokens():
    # <|python_tag|>/<|eom_id|> are special tokens in the Llama 3 tokenizer;
    # stripping them would erase the very markers this parser matches on.
    tokenizer = _RecordingTokenizer(
        '<|python_tag|>{"name": "search", "parameters": {"query": "faiss"}}<|eom_id|>'
    )
    parser = ToolParser.get_tool_parser("llama3", tokenizer)

    content, calls = await parser.extract_tool_calls([1, 2, 3])

    assert tokenizer.calls == [{"skip_special_tokens": False}]
    assert [c.name for c in calls] == ["search"]
    assert content == ""
