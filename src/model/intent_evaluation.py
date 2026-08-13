"""Scoring helpers for offline intent-routing evaluation.

The threshold-selection and promotion-gate half of this module belonged to
the retired trained classifier and went with it. What remains is what the
canonical-index instrument (``intent_index_eval``) scores with: a validated
prediction record, argmax accuracy over hand-authored queries, and the
multi-label module metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .intent_taxonomy import INTENT_LABELS, SEMANTIC_MODULES


@dataclass(frozen=True)
class IntentPredictionRecord:
    """One labeled intent prediction and its routing metadata."""

    example_id: str
    expected: str
    predicted: str
    confidence: float
    latency_ms: float
    mechanism: str


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


@dataclass(frozen=True)
class ModulePredictionRecord:
    """One query's gold and predicted module sets, plus its route outcome."""

    example_id: str
    expected: tuple[str, ...]
    predicted: tuple[str, ...]
    route_correct: bool


def module_metrics_report(
    records: Iterable[ModulePredictionRecord],
) -> dict[str, Any]:
    """Per-module precision/recall/F1, macro-F1, and joint accuracy.

    Only the thirteen semantic modules are scored. ``bare_entity`` names an
    utterance form rather than an intent, and averaging it in would distort the
    macro number. Queries with no gold modules — the original thirty predate the
    taxonomy — are excluded from scoring and counted separately, so the report
    never implies coverage it does not have.
    """
    records = tuple(records)
    scored = tuple(record for record in records if record.expected)

    per_module: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for module in SEMANTIC_MODULES:
        true_positive = sum(
            1 for r in scored if module in r.expected and module in r.predicted
        )
        false_positive = sum(
            1 for r in scored if module not in r.expected and module in r.predicted
        )
        false_negative = sum(
            1 for r in scored if module in r.expected and module not in r.predicted
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_module[module] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": true_positive + false_negative,
        }
        f1_values.append(f1)

    joint = sum(
        1 for r in scored if r.route_correct and set(r.expected) == set(r.predicted)
    )
    return {
        "total_queries": len(records),
        "scored_queries": len(scored),
        "per_module_metrics": per_module,
        "macro_f1": sum(f1_values) / len(SEMANTIC_MODULES),
        "joint_accuracy": joint / len(scored) if scored else 0.0,
    }
