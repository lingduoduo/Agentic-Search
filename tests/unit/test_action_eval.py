"""Unit tests for the action-policy eval aggregation/reporting (T-D.1 scaffold).

Aggregates the action-mix metrics that the loop already puts on
``output.metrics`` into a baseline-vs-trained comparison, so the spec's headline
success check (fewer search rounds at equal-or-better correctness) is ready the
moment a trained checkpoint exists.
"""

from __future__ import annotations

import pytest

from src.training.eval.action_eval import (
    aggregate_action_metrics,
    compare_action_evals,
    format_comparison_table,
)


def _sample(correctness, *, web=0.0, vdb=0.0, rounds=0.0, rerank=0.0, evidence=0.0):
    return {
        "correctness": correctness,
        "metrics": {
            "web_searches": web,
            "vdb_searches": vdb,
            "search_rounds": rounds,
            "rerank_calls": rerank,
            "evidence_score_final": evidence,
        },
    }


def test_aggregate_computes_means_and_mix():
    agg = aggregate_action_metrics(
        [
            _sample(1.0, web=1.0, vdb=1.0, rounds=2.0, rerank=1.0, evidence=0.8),
            _sample(0.0, web=0.0, vdb=2.0, rounds=2.0, rerank=0.0, evidence=0.4),
        ]
    )

    assert agg.n == 2
    assert agg.mean_correctness == 0.5
    assert agg.mean_search_rounds == 2.0
    assert agg.mean_web_searches == 0.5
    assert agg.mean_vdb_searches == 1.5
    assert agg.web_fraction == 1.0 / 4.0  # 1 web of 4 total searches
    assert agg.rerank_rate == 0.5  # 1 of 2 samples reranked
    assert agg.mean_evidence_score_final == pytest.approx(0.6)


def test_aggregate_empty_is_all_zero():
    agg = aggregate_action_metrics([])
    assert agg.n == 0
    assert agg.mean_search_rounds == 0.0
    assert agg.web_fraction == 0.0


def test_compare_meets_success_when_fewer_rounds_and_correctness_preserved():
    baseline = aggregate_action_metrics([_sample(1.0, vdb=4.0, rounds=4.0)])
    trained = aggregate_action_metrics([_sample(1.0, web=1.0, vdb=1.0, rounds=2.0)])

    comp = compare_action_evals(baseline, trained)

    assert comp.fewer_rounds is True
    assert comp.correctness_preserved is True
    assert comp.meets_success_criterion is True


def test_compare_fails_when_trained_uses_more_rounds():
    baseline = aggregate_action_metrics([_sample(1.0, vdb=2.0, rounds=2.0)])
    trained = aggregate_action_metrics([_sample(1.0, vdb=5.0, rounds=5.0)])

    comp = compare_action_evals(baseline, trained)

    assert comp.fewer_rounds is False
    assert comp.meets_success_criterion is False


def test_compare_fails_when_correctness_regresses():
    baseline = aggregate_action_metrics([_sample(1.0, vdb=4.0, rounds=4.0)])
    trained = aggregate_action_metrics([_sample(0.5, vdb=1.0, rounds=1.0)])

    comp = compare_action_evals(baseline, trained)

    assert comp.correctness_preserved is False
    assert comp.meets_success_criterion is False


def test_format_comparison_table_is_readable_string():
    baseline = aggregate_action_metrics([_sample(1.0, vdb=4.0, rounds=4.0)])
    trained = aggregate_action_metrics([_sample(1.0, web=1.0, vdb=1.0, rounds=2.0)])

    table = format_comparison_table(compare_action_evals(baseline, trained))

    assert isinstance(table, str)
    assert "search_rounds" in table
    assert "correctness" in table
