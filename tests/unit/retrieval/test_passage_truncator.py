from __future__ import annotations

from src.internal.retrieval.passage_truncator import PassageTruncator


def test_truncate_below_limit_unchanged():
    t = PassageTruncator(max_tokens=10)
    assert t.truncate("hello world") == "hello world"


def test_truncate_above_limit():
    t = PassageTruncator(max_tokens=3)
    result = t.truncate("one two three four five")
    assert result == "one two three"


def test_truncate_exactly_at_limit():
    t = PassageTruncator(max_tokens=3)
    assert t.truncate("a b c") == "a b c"


def test_truncate_zero_disabled():
    t = PassageTruncator(max_tokens=0)
    long = " ".join(str(i) for i in range(1000))
    assert t.truncate(long) == long


def test_truncate_empty_string():
    t = PassageTruncator(max_tokens=5)
    assert t.truncate("") == ""


def test_from_env_reads_max_tokens(monkeypatch):
    monkeypatch.setenv("RERANKER_MAX_TOKENS", "100")
    t = PassageTruncator.from_env()
    assert t._max == 100


def test_from_env_default(monkeypatch):
    monkeypatch.delenv("RERANKER_MAX_TOKENS", raising=False)
    t = PassageTruncator.from_env()
    assert t._max == 512
