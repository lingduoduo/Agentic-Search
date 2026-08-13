"""Score a built index against the held-out query sets.

Reuses the existing report machinery: IntentPredictionRecord and the metric
functions in intent_evaluation take prediction records rather than a model, so
none of it needed to change to score a different kind of model.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .intent_data import load_intent_eval_queries, load_out_of_scope_probes
from .intent_encoder import encode_texts
from .intent_evaluation import (
    IntentPredictionRecord,
    ModulePredictionRecord,
    module_metrics_report,
    realistic_accuracy_report,
)
from .intent_index_cli import check_leakage
from .intent_knn import INDEX_FILENAME, TOP_K, IntentIndex

# Legacy ids are `eval-<route>-NN`; queries added later use `bulk-NNN`. The
# legacy 30 were iterated against during canonical-set curation, so they are
# contaminated; the bulk-prefixed queries are the clean, honest slice.
LEGACY_PREFIX = "eval-"

# The clean-slice threshold sweep, fixed by the spec: never touch the legacy
# or hard slices while choosing a serving threshold.
_SWEEP_MIN_CONFIDENCES = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
_SWEEP_MIN_MARGINS = (0.02, 0.05, 0.08, 0.12)
_MIN_COVERAGE = 0.60


def _predict(index: IntentIndex, queries, thresholds):
    """Score *queries* one at a time, timing encode + decide together.

    Encoding per record (rather than batching all of them up front) costs
    throughput, but ``latency_ms`` on each ``IntentPredictionRecord`` then
    reflects true end-to-end serving latency. Batch-encoding ahead of the
    timer would make each record ~40x faster than a real request, and that
    number would be silently wrong if anything ever fed these records to
    ``evaluate_intent_predictions``, which computes p50/p95 from exactly this
    field.
    """
    records, modules, vectors = [], [], []
    for query in queries:
        start = perf_counter()
        vector = encode_texts([query.text])[0]
        decision = index.decide(vector, **thresholds)
        latency_ms = (perf_counter() - start) * 1_000
        vectors.append(vector)
        records.append(
            IntentPredictionRecord(
                example_id=query.id,
                expected=query.label,
                predicted=decision.route,
                confidence=decision.confidence,
                latency_ms=latency_ms,
                mechanism="model",
            )
        )
        modules.append(
            ModulePredictionRecord(
                example_id=query.id,
                expected=tuple(query.modules),
                predicted=tuple(decision.modules),
                route_correct=decision.route == query.label,
            )
        )
    return records, modules, np.stack(vectors) if vectors else np.empty((0,))


def _raise_on_leakage(index: IntentIndex, queries, vectors: np.ndarray) -> None:
    """Fail loudly if any *queries* duplicate a canonical example.

    Applied to every query set scored against the index, not only the bulk
    set: with nearest-neighbor routing the index *is* the model, so a leak
    anywhere manufactures accuracy on that slice the same way.
    """
    leaks = check_leakage(index, [q.text for q in queries], vectors)
    if leaks:
        raise ValueError(
            f"{len(leaks)} evaluation queries leak into the canonical set; "
            f"first: {leaks[0]}"
        )


def _argmax_report(records) -> dict[str, Any]:
    """Argmax accuracy at threshold 0.0, with the vacuous fields dropped.

    ``realistic_accuracy_report`` also returns ``coverage`` and
    ``covered_accuracy`` for a *served* threshold. At threshold 0.0 every
    record clears it by construction, so both are always 1.0 and
    ``covered_accuracy`` always equals ``accuracy`` -- reporting them here
    would read as real serving coverage to a later grep, when it is not.
    """
    report = dict(realistic_accuracy_report(records, threshold=0.0))
    report.pop("coverage", None)
    report.pop("covered_accuracy", None)
    return report


def leave_one_out_route_accuracy(index: IntentIndex) -> dict[str, Any]:
    """Score every canonical example against only the *other* examples.

    For each example, its own row is excluded from every route's candidate
    pool before taking the top-3-mean argmax, so an example can never vote
    for its own label. This measures the anchor set's self-consistency with
    no curation-target contamination at all, which makes it a better
    predictor of unseen in-domain traffic than any eval-set number.
    """
    vectors = index.vectors
    routes = [example.route for example in index.examples]
    route_row_idx = {
        route: np.array([i for i, r in enumerate(routes) if r == route])
        for route in set(routes)
    }
    similarities = vectors @ vectors.T

    correct = 0
    for position in range(len(routes)):
        best_route: str | None = None
        best_score = -2.0
        for route, rows in route_row_idx.items():
            rows_without_self = rows[rows != position]
            if rows_without_self.size == 0:
                continue
            row_similarities = similarities[position, rows_without_self]
            if row_similarities.size > TOP_K:
                row_similarities = np.partition(row_similarities, -TOP_K)[-TOP_K:]
            score = float(row_similarities.mean())
            if score > best_score:
                best_score, best_route = score, route
        if best_route == routes[position]:
            correct += 1

    return {"accuracy": correct / len(routes) if routes else 0.0, "n": len(routes)}


def _select_thresholds(
    index: IntentIndex,
    clean: tuple[IntentPredictionRecord, ...],
    clean_vectors: np.ndarray,
    probe_vectors: np.ndarray,
) -> dict[str, Any]:
    """Sweep (min_confidence, min_margin) on the clean slice only.

    Tuned exclusively against clean_151 and the out-of-scope probes -- the
    legacy_30 and hard slices are never consulted. Selects the pair with the
    highest served accuracy at coverage >= 0.60, breaking ties toward higher
    out-of-scope deferral: the rule fixed in advance by the spec.
    """
    sweep: list[dict[str, Any]] = []
    for min_confidence in _SWEEP_MIN_CONFIDENCES:
        for min_margin in _SWEEP_MIN_MARGINS:
            thresholds = {
                "min_confidence": min_confidence,
                "min_margin": min_margin,
                "min_module_score": 0.45,
            }
            served = correct = 0
            for record, vector in zip(clean, clean_vectors):
                decision = index.decide(vector, **thresholds)
                if not decision.abstained:
                    served += 1
                    correct += decision.route == record.expected
            deferred = sum(
                index.decide(vector, **thresholds).abstained for vector in probe_vectors
            )
            coverage = served / len(clean) if clean else 0.0
            sweep.append(
                {
                    "min_confidence": min_confidence,
                    "min_margin": min_margin,
                    "coverage": coverage,
                    "served_accuracy": correct / served if served else 0.0,
                    "served": served,
                    "oos_deferral": deferred / len(probe_vectors)
                    if len(probe_vectors)
                    else 0.0,
                }
            )

    eligible = [row for row in sweep if row["coverage"] >= _MIN_COVERAGE]
    eligible.sort(key=lambda row: (-row["served_accuracy"], -row["oos_deferral"]))
    return {"sweep": sweep, "selected": eligible[0] if eligible else None}


def run_index_evaluation(
    *,
    index_path: Path,
    eval_queries_path: Path,
    hard_queries_path: Path | None,
    out_of_scope_path: Path | None,
    canonical_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Score the index and write the evaluation report."""
    index = IntentIndex.load(index_path / INDEX_FILENAME)
    thresholds = {
        "min_confidence": 0.0,
        "min_margin": 0.0,
        "min_module_score": 0.45,
    }

    bulk = load_intent_eval_queries(eval_queries_path)
    bulk_records, bulk_modules, bulk_vectors = _predict(index, bulk, thresholds)
    _raise_on_leakage(index, bulk, bulk_vectors)

    legacy = tuple(r for r in bulk_records if r.example_id.startswith(LEGACY_PREFIX))
    # The clean slice: bulk-prefixed queries the canonical set was never
    # iterated against. This is the honest accuracy number.
    clean = tuple(r for r in bulk_records if not r.example_id.startswith(LEGACY_PREFIX))
    clean_vectors = np.stack(
        [
            vector
            for record, vector in zip(bulk_records, bulk_vectors)
            if not record.example_id.startswith(LEGACY_PREFIX)
        ]
    )
    clean_modules = tuple(
        m for m in bulk_modules if not m.example_id.startswith(LEGACY_PREFIX)
    )
    report: dict[str, Any] = {
        "index": {
            "size": index.size,
            "encoder": index.encoder,
            "fingerprint": index.fingerprint,
            "canonical": str(canonical_path),
            "low_support_modules": list(index.low_support_modules()),
        },
        "bulk": _argmax_report(bulk_records),
        "legacy_30": _argmax_report(legacy),
        "clean_151": _argmax_report(clean),
        "modules": module_metrics_report(bulk_modules),
        "clean_modules": module_metrics_report(clean_modules),
        "leave_one_out": leave_one_out_route_accuracy(index),
    }

    if hard_queries_path is not None:
        hard = load_intent_eval_queries(hard_queries_path)
        hard_records, hard_modules, hard_vectors = _predict(index, hard, thresholds)
        _raise_on_leakage(index, hard, hard_vectors)
        report["hard"] = _argmax_report(hard_records)
        report["hard_modules"] = module_metrics_report(hard_modules)

    if out_of_scope_path is not None:
        probes = load_out_of_scope_probes(out_of_scope_path)
        probe_vectors = encode_texts([text for _, text in probes])
        probe_confidences = [
            index.decide(vector, **thresholds).confidence for vector in probe_vectors
        ]
        # The clean slice, not the full (partially contaminated) bulk set: the
        # separation margin is only honest when measured against the queries
        # the canonical set was never iterated against.
        in_scope = [record.confidence for record in clean]
        report["out_of_scope"] = {
            "probes": len(probes),
            "mean_in_scope_confidence": sum(in_scope) / len(in_scope),
            "mean_out_of_scope_confidence": (
                sum(probe_confidences) / len(probe_confidences)
            ),
            "separation_margin": (
                sum(in_scope) / len(in_scope)
                - sum(probe_confidences) / len(probe_confidences)
            ),
            "max_out_of_scope_confidence": max(probe_confidences),
            "min_in_scope_confidence": min(in_scope),
        }
        report["threshold_tuning"] = _select_thresholds(
            index, clean, clean_vectors, probe_vectors
        )

    selected = report.get("threshold_tuning", {}).get("selected")
    report["headline"] = {
        "bulk_accuracy": report["bulk"]["accuracy"],
        "legacy_30_accuracy": report["legacy_30"]["accuracy"],
        "clean_151_accuracy": report["clean_151"]["accuracy"],
        "hard_accuracy": report.get("hard", {}).get("accuracy"),
        "module_macro_f1": report["modules"]["macro_f1"],
        "joint_accuracy": report["modules"]["joint_accuracy"],
        "separation_margin": report.get("out_of_scope", {}).get("separation_margin"),
        "leave_one_out_accuracy": report["leave_one_out"]["accuracy"],
        "selected_min_confidence": selected["min_confidence"] if selected else None,
        "selected_min_margin": selected["min_margin"] if selected else None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
