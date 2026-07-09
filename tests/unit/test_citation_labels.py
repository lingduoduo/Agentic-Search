"""Unit tests for the centralized citation-label contract."""

from __future__ import annotations

from src.context.search import (
    AgentContext,
    SearchContext,
    SearchResult,
    _citation_keys,
    citation_key,
    citation_prefix,
)


def test_prefix_and_key_format():
    assert citation_prefix(1, 2) == "R1Q2D"
    assert citation_key(1, 2, 3) == "R1Q2D3"


def test_key_is_prefix_plus_doc():
    for r, q, d in [(1, 1, 1), (2, 3, 4), (10, 2, 7)]:
        assert citation_key(r, q, d) == citation_prefix(r, q) + str(d)


def test_round_trip_parse_recovers_key():
    key = citation_key(1, 2, 3)
    assert _citation_keys(f"grounded [{key}] here") == {key}


def test_formatter_and_parser_agree_end_to_end():
    ctx = SearchContext(
        query="voice actor",
        results=[SearchResult(contents='"Voice"\nAlice David')],
    )
    # Round 1, query 1 — matches the ctx's position in agent_ctx.rounds below.
    block = ctx.to_information_block(citation_prefix=citation_prefix(1, 1))
    assert "[R1Q1D1]" in block

    agent_ctx = AgentContext(tasks={})
    agent_ctx.rounds.append([ctx])
    agent_ctx.turns.append(ctx)
    assert agent_ctx.cited_result_ids("cite [R1Q1D1]") == frozenset({"R1Q1D1"})
