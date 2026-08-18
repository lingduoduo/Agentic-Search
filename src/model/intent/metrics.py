"""Metrics over intent prediction records.

Every function here takes prediction *records* rather than a model, which is
why none of it needed to change when routing moved from a trained classifier to
the canonical-example index. ``evaluation`` is the only consumer.

This module was roughly twice this size until the retired classifier's
machinery was removed: ``evaluate_intent_predictions``,
``IntentEvaluationReport``, ``select_confidence_threshold``,
``calibration_report``, ``out_of_scope_abstention_rate``,
``compose_candidate_cascade``, ``authoritative_routes_match``,
``compare_for_promotion``, ``PromotionCriteria`` and ``PromotionDecision`` all
lost their only caller when the trained classifier and its trainer were retired
in #511, and the index harness never picked them up — it reimplemented its own
threshold sweep in ``evaluation`` instead. They were retained across #511 and
#512 pending a reviewed decision, which is this one. The promotion checklist
that #530 pre-registered is prose scored against the harness headline, and
never called ``compare_for_promotion``. Git history has all of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .model import INTENT_LABELS, SEMANTIC_MODULES


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


def separability_report(
    in_scope: Sequence[float], out_of_scope: Sequence[float]
) -> dict[str, Any]:
    """How well in-scope and out-of-scope confidences separate.

    Reported scale-free, because raw margin is not comparable across encoders.
    e5 compresses cosine similarities into a narrow high band: measured against
    the same anchors, e5-base-v2 scores a *smaller* raw margin than MiniLM
    (0.0401 vs 0.1188) while being clearly better separated (AUC 0.927 vs
    0.868). A bar in raw cosine units would reject the better model, so the bar
    is AUC and the raw margin is reported as encoder-specific context only.
    (Those two AUCs are e5-*base* against MiniLM. The shipped encoder is
    e5-*small*-v2, which measures 0.855 on the split -- compression is not the
    same thing as better separation, and only the scale-free number can tell
    you which one you got.)

    Caveat on ``cohens_d``: the pooled SD below averages each group's
    *population* variance (divided by n), not the textbook (n-1)-weighted
    form. Values from this function are therefore comparable to each other but
    not to a Cohen's d computed by scipy or any stats package.
    """
    in_scope = tuple(float(value) for value in in_scope)
    out_of_scope = tuple(float(value) for value in out_of_scope)
    if not in_scope or not out_of_scope:
        raise ValueError("separability needs a non-empty group on both sides")

    in_mean = sum(in_scope) / len(in_scope)
    out_mean = sum(out_of_scope) / len(out_of_scope)

    def _variance(values: tuple[float, ...], mean: float) -> float:
        return sum((value - mean) ** 2 for value in values) / max(len(values), 1)

    pooled = (
        (_variance(in_scope, in_mean) + _variance(out_of_scope, out_mean)) / 2
    ) ** 0.5
    labels = [1] * len(in_scope) + [0] * len(out_of_scope)
    return {
        "auc": float(roc_auc_score(labels, list(in_scope) + list(out_of_scope))),
        "cohens_d": float((in_mean - out_mean) / pooled) if pooled else 0.0,
        "raw_margin": in_mean - out_mean,
        "max_out_of_scope": max(out_of_scope),
        "min_in_scope": min(in_scope),
        "counts": {"in_scope": len(in_scope), "out_of_scope": len(out_of_scope)},
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
