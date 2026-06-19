"""Tests for QueryOptimizer expansion and spell correction."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


from src.internal.retrieval.query_optimizer import QueryOptimizer


def _write_acronyms(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


def test_expand_replaces_known_acronym(tmp_path):
    acr = tmp_path / "acronyms.json"
    _write_acronyms(
        str(acr), {"ML": "machine learning", "NLP": "natural language processing"}
    )
    opt = QueryOptimizer(str(acr), spell_enabled=False)
    assert "machine learning" in opt.expand("ML models")


def test_expand_preserves_unknown_terms(tmp_path):
    acr = tmp_path / "acronyms.json"
    _write_acronyms(str(acr), {})
    opt = QueryOptimizer(str(acr), spell_enabled=False)
    assert "procurement" in opt.expand("procurement policy")


def test_max_terms_limits_expansion(tmp_path):
    acr = tmp_path / "acronyms.json"
    _write_acronyms(str(acr), {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"})
    opt = QueryOptimizer(str(acr), max_terms=2, spell_enabled=False)
    result = opt.expand("A B C D")
    original_tokens = set("A B C D".split())
    extra = [t for t in result.split() if t not in original_tokens]
    assert len(extra) <= 2


def test_expand_returns_original_when_no_acronym_file():
    opt = QueryOptimizer(None, spell_enabled=False)
    assert opt.expand("what is FAISS") == "what is FAISS"


def test_from_env_disabled_returns_passthrough(monkeypatch):
    monkeypatch.delenv("QUERY_EXPANSION_ENABLED", raising=False)
    monkeypatch.delenv("SPELL_CORRECTION_ENABLED", raising=False)
    opt = QueryOptimizer.from_env()
    assert opt.expand("some query") == "some query"


def test_spell_correction_fixes_typo(tmp_path):
    acr = tmp_path / "acronyms.json"
    _write_acronyms(str(acr), {})
    sym = MagicMock()
    sym.lookup_compound.return_value = [MagicMock(term="retrieval")]
    with patch(
        "src.internal.retrieval.query_optimizer._build_symspell", return_value=sym
    ):
        opt = QueryOptimizer(str(acr), spell_enabled=True)
        result = opt.expand("retreival")
    assert "retrieval" in result
