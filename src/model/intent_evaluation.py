"""Evaluation, threshold selection, and promotion gates for intent models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .intent_classifier import INTENT_LABELS


@dataclass(frozen=True)
class IntentPredictionRecord:
    """One labeled intent prediction and its routing metadata."""

    example_id: str
    expected: str
    predicted: str
    confidence: float
    latency_ms: float
    mechanism: str


@dataclass(frozen=True)
class IntentEvaluationReport:
    """JSON-ready summary of a labeled intent-model evaluation."""

    threshold: float
    labels: tuple[str, ...]
    accuracy: float
    macro_f1: float
    per_label_metrics: Mapping[str, Mapping[str, float]]
    confusion_matrix: tuple[tuple[int, ...], ...]
    coverage: float
    error_rate: float
    high_confidence_errors: int
    fallback_rate: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    total_records: int
    covered_records: int
    authoritative_routes_unchanged: bool = True
    out_of_scope_abstention: float | None = None
    model_tool_precision: float | None = None

    @property
    def tool_precision(self) -> float:
        """Precision for the action-oriented ``tool`` route."""
        return self.per_label_metrics["tool"]["precision"]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this report."""
        return {
            "threshold": self.threshold,
            "labels": list(self.labels),
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_label_metrics": {
                label: dict(metrics)
                for label, metrics in self.per_label_metrics.items()
            },
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "coverage": self.coverage,
            "error_rate": self.error_rate,
            "high_confidence_errors": self.high_confidence_errors,
            "fallback_rate": self.fallback_rate,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "total_records": self.total_records,
            "covered_records": self.covered_records,
            "authoritative_routes_unchanged": self.authoritative_routes_unchanged,
            "out_of_scope_abstention": self.out_of_scope_abstention,
            "model_tool_precision": self.model_tool_precision,
        }


@dataclass(frozen=True)
class PromotionCriteria:
    """Safety and operational requirements for serving a candidate model.

    Out-of-scope abstention is deliberately absent: it is reported on the
    evaluation report, but this model family cannot reach a useful abstention
    rate at any threshold that leaves coverage. Out-of-scope safety comes from
    the LLM-classifier fallback and the clarification path, not the model.
    """

    min_tool_precision: float = 0.95
    max_high_confidence_errors: int = 0
    require_macro_f1_non_decreasing: bool = True
    require_llm_fallback_reduction: bool = True
    require_latency_improvement: bool = True


@dataclass(frozen=True)
class PromotionDecision:
    """The evaluated promotion gates and their aggregate result."""

    promotable: bool
    gates: tuple[dict[str, object], ...]
    failed_gates: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "promotable": self.promotable,
            "gates": [dict(gate) for gate in self.gates],
            "failed_gates": list(self.failed_gates),
        }


def evaluate_intent_predictions(
    records: Iterable[IntentPredictionRecord],
    *,
    threshold: float,
    authoritative_routes_unchanged: bool = True,
    out_of_scope_abstention: float | None = None,
) -> IntentEvaluationReport:
    """Calculate all metrics for labeled records at a selected threshold."""
    records = _validated_records(records)
    _validate_probability(threshold, name="threshold")

    expected = [record.expected for record in records]
    predicted = [record.predicted for record in records]
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected,
        predicted,
        labels=INTENT_LABELS,
        zero_division=0,
    )
    per_label_metrics = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, label in enumerate(INTENT_LABELS)
    }
    covered = tuple(
        record
        for record in records
        if record.mechanism == "model" and record.confidence >= threshold
    )
    high_confidence_errors = sum(
        record.expected != record.predicted for record in covered
    )
    covered_count = len(covered)
    total_records = len(records)
    coverage = covered_count / total_records
    latencies = (
        sorted(record.latency_ms for record in covered)
        if covered
        else sorted(
            record.latency_ms for record in records if record.mechanism == "classifier"
        )
    )
    fallback_count = sum(
        record.mechanism == "classifier"
        or (record.mechanism == "model" and record.confidence < threshold)
        for record in records
    )
    covered_tool_predictions = sum(record.predicted == "tool" for record in covered)
    model_tool_precision = (
        sum(
            record.predicted == "tool" and record.expected == "tool"
            for record in covered
        )
        / covered_tool_predictions
        if covered_tool_predictions
        else None
    )

    return IntentEvaluationReport(
        threshold=threshold,
        labels=tuple(INTENT_LABELS),
        accuracy=float(accuracy_score(expected, predicted)),
        macro_f1=float(sum(f1) / len(INTENT_LABELS)),
        per_label_metrics=per_label_metrics,
        confusion_matrix=tuple(
            tuple(int(value) for value in row)
            for row in confusion_matrix(expected, predicted, labels=INTENT_LABELS)
        ),
        coverage=coverage,
        error_rate=high_confidence_errors / covered_count if covered_count else 0.0,
        high_confidence_errors=high_confidence_errors,
        fallback_rate=fallback_count / total_records,
        p50_latency_ms=_percentile(latencies, 50) if latencies else None,
        p95_latency_ms=_percentile(latencies, 95) if latencies else None,
        total_records=total_records,
        covered_records=covered_count,
        authoritative_routes_unchanged=authoritative_routes_unchanged,
        out_of_scope_abstention=out_of_scope_abstention,
        model_tool_precision=model_tool_precision,
    )


def realistic_accuracy_report(
    records: Iterable[IntentPredictionRecord], *, threshold: float
) -> dict[str, Any]:
    """Score hand-authored queries the generator never produced.

    Accuracy is over every query, using the model's argmax, so the number stays
    comparable with the hand-scored diagnosis baseline. Coverage and covered
    accuracy then show what survives the serving threshold.
    """
    records = _validated_records(records)
    _validate_probability(threshold, name="threshold")

    expected = [record.expected for record in records]
    predicted = [record.predicted for record in records]
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected, predicted, labels=INTENT_LABELS, zero_division=0
    )
    covered = tuple(record for record in records if record.confidence >= threshold)
    return {
        "threshold": threshold,
        "total_queries": len(records),
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(sum(f1) / len(INTENT_LABELS)),
        "per_label_metrics": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, label in enumerate(INTENT_LABELS)
        },
        "coverage": len(covered) / len(records),
        "covered_accuracy": (
            sum(record.expected == record.predicted for record in covered)
            / len(covered)
            if covered
            else None
        ),
    }


def compose_candidate_cascade(
    model_records: Iterable[IntentPredictionRecord],
    baseline_records: Iterable[IntentPredictionRecord],
    *,
    threshold: float,
) -> tuple[IntentPredictionRecord, ...]:
    """Compose regex -> covered model -> captured fallback for held-out records."""
    _validate_probability(threshold, name="threshold")
    model_by_id = {
        record.example_id: record for record in _validated_records(model_records)
    }
    baseline_records = _validated_records(baseline_records)
    if set(model_by_id) != {record.example_id for record in baseline_records}:
        raise ValueError("Model and baseline prediction IDs must match exactly")

    candidate: list[IntentPredictionRecord] = []
    for baseline in baseline_records:
        if baseline.mechanism == "regex":
            candidate.append(baseline)
            continue
        model = model_by_id[baseline.example_id]
        candidate.append(model if model.confidence >= threshold else baseline)
    return tuple(candidate)


def authoritative_routes_match(
    candidate_records: Iterable[IntentPredictionRecord],
    baseline_records: Iterable[IntentPredictionRecord],
) -> bool:
    """Return whether every authoritative baseline route is unchanged."""
    candidate_by_id = {
        record.example_id: record for record in _validated_records(candidate_records)
    }
    baseline_records = _validated_records(baseline_records)
    authoritative = tuple(
        record for record in baseline_records if record.mechanism == "regex"
    )
    return all(
        candidate_by_id.get(record.example_id) == record for record in authoritative
    )


def select_confidence_threshold(
    records: Iterable[IntentPredictionRecord],
    *,
    tool_precision_min: float,
    max_high_confidence_errors: int,
    out_of_scope_confidences: Iterable[float] = (),
    min_out_of_scope_abstention: float = 0.0,
) -> float:
    """Choose the lowest validation threshold that satisfies route safety gates.

    When out-of-scope probe confidences are supplied, a threshold also has to
    defer at least ``min_out_of_scope_abstention`` of them, which raises the
    threshold above the confidence the model assigns to requests it cannot
    serve at all.
    """
    records = _validated_records(records)
    _validate_probability(tool_precision_min, name="tool_precision_min")
    _validate_probability(
        min_out_of_scope_abstention, name="min_out_of_scope_abstention"
    )
    if max_high_confidence_errors < 0:
        raise ValueError("max_high_confidence_errors must be non-negative")
    out_of_scope = _validated_confidences(out_of_scope_confidences)

    eligible: list[float] = []
    for threshold in sorted({record.confidence for record in records} | {1.0}):
        if (
            out_of_scope_abstention_rate(out_of_scope, threshold)
            < min_out_of_scope_abstention
        ):
            continue
        covered = tuple(record for record in records if record.confidence >= threshold)
        if not covered:
            # The explicit 1.0 candidate safely defers every sub-1.0 prediction.
            eligible.append(threshold)
            continue
        report = evaluate_intent_predictions(covered, threshold=0.0)
        if (
            report.tool_precision >= tool_precision_min
            and report.high_confidence_errors <= max_high_confidence_errors
        ):
            eligible.append(threshold)

    if not eligible:
        raise ValueError(
            "No confidence threshold satisfies the requested safety limits"
        )
    return min(eligible)


def calibration_report(
    records: Iterable[IntentPredictionRecord],
    *,
    selected_threshold: float,
    out_of_scope_confidences: Iterable[float] = (),
    bins: int = 10,
) -> dict[str, Any]:
    """Report how confidence relates to correctness, not just the chosen cut.

    Softmax scores are not probabilities, so the sweep and reliability bins let
    an operator see the coverage/error trade at every candidate threshold
    instead of trusting the single selected one.
    """
    records = _validated_records(records)
    _validate_probability(selected_threshold, name="selected_threshold")
    if bins < 1:
        raise ValueError("bins must be positive")
    out_of_scope = _validated_confidences(out_of_scope_confidences)

    sweep: list[dict[str, Any]] = []
    for threshold in sorted({record.confidence for record in records} | {1.0}):
        covered = tuple(record for record in records if record.confidence >= threshold)
        row: dict[str, Any] = {
            "threshold": threshold,
            "coverage": len(covered) / len(records),
            "covered_records": len(covered),
            "out_of_scope_abstention": (
                out_of_scope_abstention_rate(out_of_scope, threshold)
                if out_of_scope
                else None
            ),
        }
        if covered:
            report = evaluate_intent_predictions(covered, threshold=0.0)
            row["macro_f1"] = report.macro_f1
            row["tool_precision"] = report.tool_precision
            row["high_confidence_errors"] = report.high_confidence_errors
        else:
            row["macro_f1"] = None
            row["tool_precision"] = None
            row["high_confidence_errors"] = 0
        sweep.append(row)

    reliability: list[dict[str, Any]] = []
    expected_calibration_error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = tuple(
            record
            for record in records
            if lower <= record.confidence < upper
            or (index == bins - 1 and record.confidence == 1.0)
        )
        if not members:
            reliability.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "accuracy": None,
                    "mean_confidence": None,
                }
            )
            continue
        accuracy = sum(record.expected == record.predicted for record in members) / len(
            members
        )
        mean_confidence = sum(record.confidence for record in members) / len(members)
        expected_calibration_error += (
            len(members) / len(records) * abs(accuracy - mean_confidence)
        )
        reliability.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
            }
        )

    return {
        "selected_threshold": selected_threshold,
        "thresholds": sweep,
        "reliability_bins": reliability,
        "expected_calibration_error": expected_calibration_error,
        "out_of_scope_probes": len(out_of_scope),
        # None, not 1.0: no probes means unmeasured, not safe.
        "out_of_scope_abstention": (
            out_of_scope_abstention_rate(out_of_scope, selected_threshold)
            if out_of_scope
            else None
        ),
    }


def out_of_scope_abstention_rate(
    confidences: tuple[float, ...], threshold: float
) -> float:
    """Fraction of out-of-scope probes the threshold declines to serve."""
    if not confidences:
        return 1.0
    return sum(value < threshold for value in confidences) / len(confidences)


def _validated_confidences(values: Iterable[float]) -> tuple[float, ...]:
    values = tuple(values)
    for value in values:
        _validate_probability(value, name="out-of-scope confidence")
    return values


def compare_for_promotion(
    candidate: IntentEvaluationReport,
    baseline: IntentEvaluationReport,
    criteria: PromotionCriteria,
) -> PromotionDecision:
    """Compare a candidate report with a baseline and report every gate."""
    _validate_probability(criteria.min_tool_precision, name="min_tool_precision")
    if criteria.max_high_confidence_errors < 0:
        raise ValueError("max_high_confidence_errors must be non-negative")

    gates = (
        _gate(
            "macro_f1_non_decreasing",
            not criteria.require_macro_f1_non_decreasing
            or candidate.macro_f1 >= baseline.macro_f1,
            candidate.macro_f1,
            baseline.macro_f1,
        ),
        _gate(
            "tool_precision_minimum",
            # Unmeasured is not evidence of safety, as with out-of-scope.
            candidate.model_tool_precision is not None
            and candidate.model_tool_precision >= criteria.min_tool_precision,
            candidate.model_tool_precision,
            criteria.min_tool_precision,
        ),
        _gate(
            "high_confidence_errors_maximum",
            candidate.high_confidence_errors <= criteria.max_high_confidence_errors,
            candidate.high_confidence_errors,
            criteria.max_high_confidence_errors,
        ),
        _relative_gate(
            "llm_fallback_reduction",
            candidate.fallback_rate,
            baseline.fallback_rate,
            criteria.require_llm_fallback_reduction,
        ),
        _relative_gate(
            "latency_improvement",
            candidate.p50_latency_ms,
            baseline.p50_latency_ms,
            criteria.require_latency_improvement,
        ),
        _gate(
            "authoritative_routes_unchanged",
            candidate.authoritative_routes_unchanged,
            int(candidate.authoritative_routes_unchanged),
            1,
        ),
    )
    failed_gates = tuple(gate["name"] for gate in gates if not gate["passed"])
    return PromotionDecision(
        promotable=not failed_gates,
        gates=gates,
        failed_gates=failed_gates,
    )


def _validated_records(
    records: Iterable[IntentPredictionRecord],
) -> tuple[IntentPredictionRecord, ...]:
    records = tuple(records)
    if not records:
        raise ValueError("At least one intent prediction record is required")
    labels = set(INTENT_LABELS)
    mechanisms = {"regex", "classifier", "rule_based", "model"}
    for record in records:
        if record.expected not in labels or record.predicted not in labels:
            raise ValueError(
                "Intent prediction records must use supported intent labels"
            )
        _validate_probability(record.confidence, name="record confidence")
        if not math.isfinite(record.latency_ms) or record.latency_ms < 0:
            raise ValueError("record latency_ms must be a non-negative finite number")
        if record.mechanism not in mechanisms:
            raise ValueError("Intent prediction mechanism is unsupported")
    return records


def _validate_probability(value: float, *, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite probability between 0 and 1")


def _percentile(values: list[float], percentile: float) -> float:
    position = (len(values) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def _gate(
    name: str, passed: bool, candidate: float | int, baseline_or_limit: float | int
) -> dict[str, object]:
    return {
        "name": name,
        "passed": passed,
        "candidate": candidate,
        "baseline_or_limit": baseline_or_limit,
    }


def _relative_gate(
    name: str,
    candidate: float | None,
    baseline: float | None,
    required: bool,
) -> dict[str, object]:
    return _gate(
        name,
        not required
        or (candidate is not None and baseline is not None and candidate < baseline),
        candidate,
        baseline,
    )
