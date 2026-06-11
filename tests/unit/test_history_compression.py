"""Tests for src/context/history.py."""

from __future__ import annotations


from src.context.history import compress_history, estimate_tokens
from src.context.models import ChatMessage


def _msg(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)


class TestEstimateTokens:
    def test_basic(self):
        assert estimate_tokens("hello") == 1  # 5 chars → max(1, 1)
        assert estimate_tokens("a" * 40) == 10  # 40 // 4 = 10

    def test_empty_returns_one(self):
        assert estimate_tokens("") == 1

    def test_long_text(self):
        assert estimate_tokens("x" * 4000) == 1000


class TestCompressHistory:
    def test_under_budget_returns_as_is(self):
        msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
        result = compress_history(msgs, None, max_tokens=10_000)
        assert result == msgs

    def test_over_budget_no_llm_drops_old(self):
        # 12 messages, each 100 chars → ~300 tokens total > max_tokens=50
        msgs = [
            _msg("user" if i % 2 == 0 else "assistant", "x" * 100) for i in range(12)
        ]
        result = compress_history(msgs, None, max_tokens=50, keep_recent=4)
        assert result == msgs[-4:]

    def test_over_budget_few_messages_returns_as_is(self):
        msgs = [_msg("user", "x" * 1000)]
        result = compress_history(msgs, None, max_tokens=10, keep_recent=6)
        assert result == msgs

    def test_over_budget_with_llm_adds_summary(self):
        class _FakeLLM:
            def complete(self, messages, **kwargs):
                class _R:
                    text = "Summary of old conversation"

                return _R()

        msgs = [
            _msg("user" if i % 2 == 0 else "assistant", "x" * 200) for i in range(10)
        ]
        result = compress_history(msgs, _FakeLLM(), max_tokens=50, keep_recent=4)
        assert result[0].role == "system"
        assert "Summary" in result[0].content
        assert len(result) == 5  # 1 summary + 4 recent

    def test_over_budget_llm_failure_falls_back_to_drop(self):
        class _BrokenLLM:
            def complete(self, messages, **kwargs):
                raise RuntimeError("LLM down")

        msgs = [
            _msg("user" if i % 2 == 0 else "assistant", "x" * 200) for i in range(10)
        ]
        result = compress_history(msgs, _BrokenLLM(), max_tokens=50, keep_recent=4)
        assert result == msgs[-4:]
