from __future__ import annotations

from unittest.mock import MagicMock

from src.internal.retrieval.multi_query import MultiQueryGenerator


def _llm(text):
    llm = MagicMock()
    llm.complete.return_value = type("R", (), {"text": text})()
    return llm


def test_parses_numbered_lines_and_caps_n():
    gen = MultiQueryGenerator(_llm("1. a\n2. b\n3. c\n4. d"), n=3)
    assert gen.generate("orig") == ["a", "b", "c"]


def test_empty_on_llm_failure():
    bad = MagicMock()
    bad.complete.side_effect = RuntimeError("boom")
    assert MultiQueryGenerator(bad).generate("q") == []
