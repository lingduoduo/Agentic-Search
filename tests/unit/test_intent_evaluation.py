import json

import pytest

from src.model.intent_evaluation import (
    calibration_report,
    IntentEvaluationReport,
    IntentPredictionRecord,
    PromotionCriteria,
    authoritative_routes_match,
    compare_for_promotion,
    compose_candidate_cascade,
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
    model_tool_precision: float | None = None,
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
        model_tool_precision=model_tool_precision,
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
            model_tool_precision=1.0,
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
            model_tool_precision=0.97,
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
        "authoritative_routes_unchanged",
        "out_of_scope_abstention_minimum",
    }
    # An unmeasured out-of-scope rate is not evidence of safety.
    assert gates["out_of_scope_abstention_minimum"]["passed"] is False
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
        "out_of_scope_abstention_minimum",
    }


def test_candidate_cascade_keeps_regex_routes_and_models_only_ambiguous_queries():
    baseline = [
        IntentPredictionRecord("regex", "search", "search", 1.0, 0.1, "regex"),
        IntentPredictionRecord("ambiguous", "tool", "chat", 1.0, 40.0, "classifier"),
    ]
    model = [
        IntentPredictionRecord("regex", "search", "tool", 0.99, 1.0, "model"),
        IntentPredictionRecord("ambiguous", "tool", "tool", 0.98, 1.0, "model"),
    ]

    candidate = compose_candidate_cascade(model, baseline, threshold=0.9)
    report = evaluate_intent_predictions(candidate, threshold=0.9)

    assert candidate[0] == baseline[0]
    assert candidate[1].mechanism == "model"
    assert report.covered_records == 1
    assert report.coverage == 0.5
    assert report.fallback_rate == 0.0


def test_candidate_cascade_abstention_uses_same_captured_fallback():
    baseline = [
        IntentPredictionRecord("ambiguous", "tool", "chat", 1.0, 40.0, "classifier")
    ]
    model = [IntentPredictionRecord("ambiguous", "tool", "tool", 0.5, 1.0, "model")]

    candidate = compose_candidate_cascade(model, baseline, threshold=0.9)

    assert candidate == tuple(baseline)


def test_authoritative_route_gate_fails_when_candidate_changes_regex_record():
    baseline = [IntentPredictionRecord("regex", "search", "search", 1.0, 0.1, "regex")]
    candidate = [IntentPredictionRecord("regex", "search", "tool", 0.99, 1.0, "model")]
    report = evaluate_intent_predictions(
        candidate,
        threshold=0.9,
        authoritative_routes_unchanged=authoritative_routes_match(candidate, baseline),
    )

    decision = compare_for_promotion(
        report,
        _report(
            macro_f1=1.0,
            tool_precision=1.0,
            fallback_rate=1.0,
            p50_latency_ms=40.0,
        ),
        PromotionCriteria(),
    )

    assert "authoritative_routes_unchanged" in decision.failed_gates


def test_tool_precision_gate_ignores_deterministic_routes():
    """Regex-decided tool routes must not dilute the model's own precision.

    Forty correct deterministic tool routes plus two false model tool routes
    give 0.9524 cascade precision, which clears the 0.95 limit while the model
    itself got every tool route wrong.
    """
    records = [
        IntentPredictionRecord(f"r{i}", "tool", "tool", 1.0, 0.1, "regex")
        for i in range(40)
    ] + [
        IntentPredictionRecord(f"m{i}", "chat", "tool", 0.99, 0.1, "model")
        for i in range(2)
    ]
    report = evaluate_intent_predictions(records, threshold=0.5)

    assert report.tool_precision == pytest.approx(0.9524, abs=1e-4)
    assert report.model_tool_precision == 0.0

    decision = compare_for_promotion(
        report,
        _report(
            macro_f1=0.0,
            tool_precision=1.0,
            fallback_rate=1.0,
            p50_latency_ms=400.0,
        ),
        PromotionCriteria(max_high_confidence_errors=2),
    )

    assert "tool_precision_minimum" in decision.failed_gates


def test_tool_precision_is_unmeasured_when_the_model_predicts_no_tool_route():
    records = [IntentPredictionRecord("m1", "chat", "chat", 0.99, 0.1, "model")]
    report = evaluate_intent_predictions(records, threshold=0.5)

    assert report.model_tool_precision is None

    decision = compare_for_promotion(
        report,
        _report(
            macro_f1=0.0,
            tool_precision=1.0,
            fallback_rate=1.0,
            p50_latency_ms=400.0,
        ),
        PromotionCriteria(),
    )

    assert "tool_precision_minimum" in decision.failed_gates


def test_threshold_selection_defers_out_of_scope_requests():
    """A threshold that serves out-of-scope probes is rejected.

    Softmax over three labels always names one, so out-of-scope safety comes
    only from requiring those probes to land below the served threshold.
    """
    records = [
        IntentPredictionRecord("1", "tool", "tool", 0.55, 1.0, "model"),
        IntentPredictionRecord("2", "tool", "tool", 0.85, 1.0, "model"),
        IntentPredictionRecord("3", "chat", "chat", 0.90, 1.0, "model"),
    ]
    without = select_confidence_threshold(
        records, tool_precision_min=0.95, max_high_confidence_errors=0
    )
    with_oos = select_confidence_threshold(
        records,
        tool_precision_min=0.95,
        max_high_confidence_errors=0,
        out_of_scope_confidences=(0.60, 0.62),
        min_out_of_scope_abstention=1.0,
    )
    assert without == 0.55
    assert with_oos == 0.85  # lifted above every out-of-scope probe


def test_calibration_report_sweeps_thresholds_and_scores_reliability():
    records = [
        IntentPredictionRecord("1", "chat", "chat", 0.90, 1.0, "model"),
        IntentPredictionRecord("2", "chat", "search", 0.55, 1.0, "model"),
    ]
    calibration = calibration_report(
        records, selected_threshold=0.90, out_of_scope_confidences=(0.40,)
    )

    thresholds = [row["threshold"] for row in calibration["thresholds"]]
    assert 0.55 in thresholds and 0.90 in thresholds
    selected = next(
        row for row in calibration["thresholds"] if row["threshold"] == 0.90
    )
    assert selected["coverage"] == 0.5
    assert selected["high_confidence_errors"] == 0
    assert selected["out_of_scope_abstention"] == 1.0
    assert 0.0 <= calibration["expected_calibration_error"] <= 1.0
    assert sum(b["count"] for b in calibration["reliability_bins"]) == 2


def test_calibration_reports_unmeasured_out_of_scope_as_none():
    """No probes means unmeasured, never a perfect abstention score."""
    calibration = calibration_report(
        [IntentPredictionRecord("1", "chat", "chat", 0.9, 1.0, "model")],
        selected_threshold=0.9,
    )

    assert calibration["out_of_scope_probes"] == 0
    assert calibration["out_of_scope_abstention"] is None
    assert all(
        row["out_of_scope_abstention"] is None for row in calibration["thresholds"]
    )
