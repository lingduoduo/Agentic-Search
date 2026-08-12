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
        }


@dataclass(frozen=True)
class PromotionCriteria:
    """Safety and operational requirements for serving a candidate model."""

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
    )


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
) -> float:
    """Choose the lowest validation threshold that satisfies route safety gates."""
    records = _validated_records(records)
    _validate_probability(tool_precision_min, name="tool_precision_min")
    if max_high_confidence_errors < 0:
        raise ValueError("max_high_confidence_errors must be non-negative")

    eligible: list[tuple[float, float, float]] = []
    for threshold in sorted({record.confidence for record in records} | {1.0}):
        covered = tuple(record for record in records if record.confidence >= threshold)
        if not covered:
            # The explicit 1.0 candidate safely defers every sub-1.0 prediction.
            eligible.append((threshold, 0.0, 0.0))
            continue
        report = evaluate_intent_predictions(covered, threshold=0.0)
        if (
            report.tool_precision >= tool_precision_min
            and report.high_confidence_errors <= max_high_confidence_errors
        ):
            eligible.append((threshold, report.macro_f1, len(covered) / len(records)))

    if not eligible:
        raise ValueError(
            "No confidence threshold satisfies the requested safety limits"
        )
    return min(
        eligible, key=lambda candidate: (candidate[0], -candidate[1], -candidate[2])
    )[0]


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
            candidate.tool_precision >= criteria.min_tool_precision,
            candidate.tool_precision,
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
