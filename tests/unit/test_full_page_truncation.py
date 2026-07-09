"""Unit tests for full-page fetch content truncation."""

from __future__ import annotations

from src.agents.search.search import (
    SearchAgentLoop,
    SearchAgentLoopConfig,
    _truncate_page_content,
)
from src.context.search import SearchResult
from tests.unit.test_agent_loop import (
    DummyServerManager,
    DummyTokenizerWithEncode,
)


def test_truncate_under_limit_unchanged():
    assert _truncate_page_content("short", 100) == "short"


def test_truncate_over_limit_head_kept_with_marker():
    text = "x" * 500
    out = _truncate_page_content(text, 100)
    assert out == "x" * 100 + "…(truncated)"
    assert out.startswith("x" * 100)
    assert out.endswith("…(truncated)")


def test_truncate_disabled_when_limit_non_positive():
    text = "y" * 500
    assert _truncate_page_content(text, 0) == text
    assert _truncate_page_content(text, -1) == text


def _loop(max_chars: int) -> SearchAgentLoop:
    return SearchAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
        search_config=SearchAgentLoopConfig(max_full_page_chars=max_chars),
    )


def test_format_full_page_truncates_large_page():
    loop = _loop(50)
    pages = [SearchResult(contents="z" * 500, title="Big", url="http://x")]
    block = loop._format_full_page_information(pages)
    assert "z" * 50 + "…(truncated)" in block
    assert "z" * 51 not in block  # content beyond the cap is gone


def test_format_full_page_keeps_small_page_intact():
    loop = _loop(4096)
    pages = [SearchResult(contents="small body", title="S", url="http://y")]
    block = loop._format_full_page_information(pages)
    assert "small body" in block
    assert "truncated" not in block


def test_default_cap_is_4096():
    assert SearchAgentLoopConfig().max_full_page_chars == 4096
