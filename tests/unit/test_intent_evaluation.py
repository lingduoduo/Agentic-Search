import json

import pytest

from src.model.intent_evaluation import (
    IntentEvaluationReport,
    IntentPredictionRecord,
    PromotionCriteria,
    compare_for_promotion,
    evaluate_intent_predictions,
    select_confidence_threshold,
)


def _report(
    *,
    macro_f1: float,
    tool_precision: float,
    fallback_rate: float | None,
    p50_latency_ms: float | None,
    high_confidence_errors: int = 0,
) -> IntentEvaluationReport:
    return IntentEvaluationReport(
        threshold=0.8,
        labels=("chat", "search", "tool"),
        accuracy=macro_f1,
        macro_f1=macro_f1,
        per_label_metrics={
            "chat": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            "search": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            "tool": {"precision": tool_precision, "recall": 1.0, "f1": 1.0},
        },
        confusion_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        coverage=1.0 if fallback_rate == 0 else 0.8,
        error_rate=0.0,
        high_confidence_errors=high_confidence_errors,
        fallback_rate=fallback_rate,
        p50_latency_ms=p50_latency_ms,
        p95_latency_ms=p50_latency_ms,
        total_records=3,
        covered_records=3,
    )


def test_threshold_selection_prioritizes_tool_precision():
    records = [
        IntentPredictionRecord("1", "chat", "tool", 0.70, 1.0, "model"),
        IntentPredictionRecord("2", "tool", "tool", 0.91, 1.0, "model"),
        IntentPredictionRecord("3", "tool", "tool", 0.95, 1.0, "model"),
        IntentPredictionRecord("4", "search", "search", 0.80, 1.0, "model"),
    ]

    threshold = select_confidence_threshold(
        records, tool_precision_min=0.95, max_high_confidence_errors=0
    )

    assert threshold > 0.70


def test_evaluation_report_includes_metrics_at_selected_threshold():
    records = [
        IntentPredictionRecord("1", "chat", "chat", 0.99, 2.0, "model"),
        IntentPredictionRecord("2", "search", "tool", 0.40, 4.0, "model"),
        IntentPredictionRecord("3", "tool", "tool", 0.80, 6.0, "model"),
    ]

    report = evaluate_intent_predictions(records, threshold=0.75)

    assert report.accuracy == pytest.approx(2 / 3)
    assert report.macro_f1 == pytest.approx(5 / 9)
    assert report.per_label_metrics["tool"]["precision"] == 0.5
    assert report.coverage == pytest.approx(2 / 3)
    assert report.error_rate == 0.0
    assert report.high_confidence_errors == 0
    assert report.fallback_rate == pytest.approx(1 / 3)
    assert report.p50_latency_ms == 4.0
    assert report.to_dict()["confusion_matrix"] == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
    json.dumps(report.to_dict())


def test_threshold_selection_rejects_unsafe_covered_records():
    records = [
        IntentPredictionRecord("1", "chat", "tool", 0.99, 1.0, "model"),
        IntentPredictionRecord("2", "tool", "tool", 1.00, 1.0, "model"),
    ]

    threshold = select_confidence_threshold(
        records, tool_precision_min=1.0, max_high_confidence_errors=0
    )

    assert threshold == 1.0


def test_threshold_selection_uses_one_as_an_abstention_sentinel_when_unsafe():
    records = [
        IntentPredictionRecord("1", "chat", "tool", 0.70, 1.0, "model"),
        IntentPredictionRecord("2", "search", "tool", 0.90, 1.0, "model"),
    ]

    threshold = select_confidence_threshold(
        records, tool_precision_min=0.95, max_high_confidence_errors=0
    )

    assert threshold == 1.0


def test_promotion_fails_when_macro_f1_regresses():
    decision = compare_for_promotion(
        candidate=_report(
            macro_f1=0.80,
            tool_precision=1.0,
            fallback_rate=0.2,
            p50_latency_ms=2.0,
        ),
        baseline=_report(
            macro_f1=0.85,
            tool_precision=1.0,
            fallback_rate=0.5,
            p50_latency_ms=50.0,
        ),
        criteria=PromotionCriteria(),
    )

    assert decision.promotable is False
    assert "macro_f1_non_decreasing" in decision.failed_gates


def test_promotion_reports_each_gate_and_rejects_missing_baseline_measurements():
    decision = compare_for_promotion(
        candidate=_report(
            macro_f1=0.90,
            tool_precision=0.97,
            fallback_rate=0.2,
            p50_latency_ms=2.0,
        ),
        baseline=_report(
            macro_f1=0.90,
            tool_precision=0.97,
            fallback_rate=None,
            p50_latency_ms=None,
        ),
        criteria=PromotionCriteria(),
    )

    gates = {gate["name"]: gate for gate in decision.gates}
    assert set(gates) == {
        "macro_f1_non_decreasing",
        "tool_precision_minimum",
        "high_confidence_errors_maximum",
        "llm_fallback_reduction",
        "latency_improvement",
    }
    assert gates["llm_fallback_reduction"] == {
        "name": "llm_fallback_reduction",
        "passed": False,
        "candidate": 0.2,
        "baseline_or_limit": None,
    }
    assert gates["latency_improvement"]["passed"] is False
    assert set(decision.failed_gates) == {
        "llm_fallback_reduction",
        "latency_improvement",
    }
