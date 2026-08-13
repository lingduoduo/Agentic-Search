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
from .intent_encoder import DEFAULT_ENCODER, encode_texts
from .intent_eval_split import DEFAULT_SEED, DEFAULT_SLICE_SIZE, split_eval_queries
from .intent_evaluation import (
    IntentPredictionRecord,
    ModulePredictionRecord,
    module_metrics_report,
    realistic_accuracy_report,
    separability_report,
)
from .intent_index_cli import _fingerprint, check_leakage
from .intent_knn import INDEX_FILENAME, TOP_K, IntentIndex, KnnDecision

# Legacy ids are `eval-<route>-NN`; queries added later use `bulk-NNN`. The
# legacy 30 were iterated against during canonical-set curation, so they are
# contaminated -- worthless as a gate and therefore free to tune on. Every
# legacy query lands in the tuning split (see intent_eval_split), which is
# what preserves the bulk-prefixed queries not drawn into that split as an
# untouched test set.
LEGACY_PREFIX = "eval-"

# The tuning-slice threshold sweep, fixed by the spec: never touch the test
# slice or hard slice while choosing serving hyperparameters.
_SWEEP_MIN_CONFIDENCES = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
_SWEEP_MIN_MARGINS = (0.02, 0.05, 0.08, 0.12)
_MIN_COVERAGE = 0.60
_SWEEP_TOP_K = (3, 5, 8, 15, 25)

# Mirrors AppSettings.intent_min_module_score's default (src/internal/configs/
# app_configs.py) so evaluation scores modules with the same bar serving uses.
_DEFAULT_MIN_MODULE_SCORE = 0.45


def _predict(index: IntentIndex, queries, thresholds, *, model_name: str):
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
        vector = encode_texts([query.text], model_name=model_name)[0]
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


def leave_one_out_route_accuracy(
    index: IntentIndex, *, top_k: int = TOP_K
) -> dict[str, Any]:
    """Score every canonical example against only the *other* examples.

    For each example, its own row is excluded from every route's candidate
    pool before taking the top-k-mean argmax, so an example can never vote
    for its own label. This measures the anchor set's self-consistency with
    no curation-target contamination at all, which makes it a useful,
    but biased, predictor: past top_k~15 it keeps climbing even as held-out
    accuracy turns down, because it is measuring the anchor set's internal
    cohesion rather than predicting held-out behaviour. It is reported
    everywhere below, but it never selects a hyperparameter.
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
            if row_similarities.size > top_k:
                row_similarities = np.partition(row_similarities, -top_k)[-top_k:]
            score = float(row_similarities.mean())
            if score > best_score:
                best_score, best_route = score, route
        if best_route == routes[position]:
            correct += 1

    return {"accuracy": correct / len(routes) if routes else 0.0, "n": len(routes)}


def _decide_batch(
    index: IntentIndex,
    vectors: np.ndarray,
    *,
    top_k: int,
    min_confidence: float,
    min_margin: float,
) -> list[KnnDecision]:
    thresholds = {
        "min_confidence": min_confidence,
        "min_margin": min_margin,
        "min_module_score": _DEFAULT_MIN_MODULE_SCORE,
        "top_k": top_k,
    }
    return [index.decide(vector, **thresholds) for vector in vectors]


def _serving_report(
    index: IntentIndex,
    records: tuple[IntentPredictionRecord, ...],
    vectors: np.ndarray,
    *,
    top_k: int,
    min_confidence: float,
    min_margin: float,
) -> dict[str, Any]:
    """Accuracy and coverage at specific serving hyperparameters.

    ``accuracy`` is the argmax route match regardless of abstention, so it
    stays comparable with the hand-scored diagnosis baseline the way
    ``realistic_accuracy_report`` is; ``coverage``/``served_accuracy`` show
    what a caller actually sees once ``min_confidence``/``min_margin`` gate
    the response.
    """
    decisions = _decide_batch(
        index,
        vectors,
        top_k=top_k,
        min_confidence=min_confidence,
        min_margin=min_margin,
    )
    total = len(records)
    argmax_correct = sum(d.route == r.expected for d, r in zip(decisions, records))
    served = [(d, r) for d, r in zip(decisions, records) if not d.abstained]
    served_correct = sum(d.route == r.expected for d, r in served)
    return {
        "total_queries": total,
        "accuracy": argmax_correct / total if total else 0.0,
        "coverage": len(served) / total if total else 0.0,
        "served_accuracy": served_correct / len(served) if served else 0.0,
        "served": len(served),
    }


def _select_thresholds(
    index: IntentIndex,
    tuning: tuple[IntentPredictionRecord, ...],
    tuning_vectors: np.ndarray,
    probe_vectors: np.ndarray,
) -> dict[str, Any]:
    """Sweep (top_k, min_confidence, min_margin) on the tuning slice only.

    Tuned exclusively against the tuning split and the out-of-scope probes --
    the test slice and hard slice are never consulted. Selects the
    combination with the highest served accuracy at coverage >= 0.60,
    breaking ties toward higher out-of-scope deferral: the rule fixed in
    advance by the spec. ``leave_one_out_route_accuracy`` is not part of this
    key -- see its docstring for why it would be a biased selector.
    """
    sweep: list[dict[str, Any]] = []
    for top_k in _SWEEP_TOP_K:
        for min_confidence in _SWEEP_MIN_CONFIDENCES:
            for min_margin in _SWEEP_MIN_MARGINS:
                decisions = _decide_batch(
                    index,
                    tuning_vectors,
                    top_k=top_k,
                    min_confidence=min_confidence,
                    min_margin=min_margin,
                )
                served = [(d, r) for d, r in zip(decisions, tuning) if not d.abstained]
                correct = sum(d.route == r.expected for d, r in served)
                probe_decisions = _decide_batch(
                    index,
                    probe_vectors,
                    top_k=top_k,
                    min_confidence=min_confidence,
                    min_margin=min_margin,
                )
                deferred = sum(d.abstained for d in probe_decisions)
                coverage = len(served) / len(tuning) if tuning else 0.0
                sweep.append(
                    {
                        "top_k": top_k,
                        "min_confidence": min_confidence,
                        "min_margin": min_margin,
                        "coverage": coverage,
                        "served_accuracy": correct / len(served) if served else 0.0,
                        "served": len(served),
                        "oos_deferral": deferred / len(probe_vectors)
                        if len(probe_vectors)
                        else 0.0,
                    }
                )

    eligible = [row for row in sweep if row["coverage"] >= _MIN_COVERAGE]
    eligible.sort(key=lambda row: (-row["served_accuracy"], -row["oos_deferral"]))
    return {"sweep": sweep, "selected": eligible[0] if eligible else None}


def _accuracy_at_top_k(
    index: IntentIndex,
    records: tuple[IntentPredictionRecord, ...],
    vectors: np.ndarray,
    top_k: int,
) -> float:
    """Argmax route accuracy over *records* at a given *top_k*, threshold 0.0."""
    if not records:
        return 0.0
    correct = sum(
        index.decide(
            vector,
            min_confidence=0.0,
            min_margin=0.0,
            min_module_score=_DEFAULT_MIN_MODULE_SCORE,
            top_k=top_k,
        ).route
        == record.expected
        for record, vector in zip(records, vectors)
    )
    return correct / len(records)


def _separation_margin_at_top_k(
    index: IntentIndex,
    test_vectors: np.ndarray,
    probe_vectors: np.ndarray,
    top_k: int,
) -> float:
    """Mean in-scope minus mean out-of-scope confidence at a given *top_k*."""
    kwargs = {
        "min_confidence": 0.0,
        "min_margin": 0.0,
        "min_module_score": _DEFAULT_MIN_MODULE_SCORE,
        "top_k": top_k,
    }
    in_scope = [index.decide(v, **kwargs).confidence for v in test_vectors]
    out_of_scope = [index.decide(v, **kwargs).confidence for v in probe_vectors]
    return sum(in_scope) / len(in_scope) - sum(out_of_scope) / len(out_of_scope)


def _sweep_top_k(
    index: IntentIndex,
    test: tuple[IntentPredictionRecord, ...],
    test_vectors: np.ndarray,
    hard: tuple[IntentPredictionRecord, ...] | None,
    hard_vectors: np.ndarray | None,
    probe_vectors: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Report-only sweep of TOP_K over the already-built index and encoder.

    Evidence for a later decision, not a selection: TOP_K stays 3 in serving
    regardless of this table. Draws on the test slice (never the tuning
    slice) so this evidence stays as honest as everything else in the report.
    See the module-level ``_SWEEP_TOP_K`` comment and
    docs/training-and-evaluation.md.
    """
    rows: list[dict[str, Any]] = []
    for top_k in _SWEEP_TOP_K:
        row: dict[str, Any] = {
            "top_k": top_k,
            "test_slice_accuracy": _accuracy_at_top_k(index, test, test_vectors, top_k),
            "leave_one_out_accuracy": leave_one_out_route_accuracy(index, top_k=top_k)[
                "accuracy"
            ],
        }
        if hard is not None and hard_vectors is not None:
            row["hard_accuracy"] = _accuracy_at_top_k(index, hard, hard_vectors, top_k)
        if probe_vectors is not None and len(probe_vectors):
            row["separation_margin"] = _separation_margin_at_top_k(
                index, test_vectors, probe_vectors, top_k
            )
        rows.append(row)
    return rows


def run_index_evaluation(
    *,
    index_path: Path,
    eval_queries_path: Path,
    hard_queries_path: Path | None,
    out_of_scope_path: Path | None,
    canonical_path: Path,
    output_path: Path,
    model_name: str = DEFAULT_ENCODER,
    slice_size: int = DEFAULT_SLICE_SIZE,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Score the index and write the evaluation report."""
    index = IntentIndex.load(index_path / INDEX_FILENAME)
    if index.encoder != model_name:
        # Both all-MiniLM-L6-v2 and e5-small-v2 are 384-dimensional, so a
        # mismatched encoder has no other symptom: no shape error, no
        # exception, just a confident, meaningless number. This is not
        # hypothetical -- it is exactly what happened once already, scoring a
        # stale MiniLM-built index against e5-encoded queries.
        raise ValueError(
            f"intent-index at {index_path} was built with encoder "
            f"{index.encoder!r}, but this evaluation is embedding queries "
            f"with {model_name!r}. Rebuild the index with the current "
            "encoder (`python -m src.model.intent_index_cli build`) before "
            "evaluating."
        )
    canonical_fingerprint = _fingerprint(canonical_path)
    if canonical_fingerprint != index.fingerprint:
        raise ValueError(
            f"canonical file {canonical_path} has fingerprint "
            f"{canonical_fingerprint!r}, but the index at {index_path} was built "
            f"from fingerprint {index.fingerprint!r}. The canonical set changed "
            "since the index was built; run `intent_index_cli build` again "
            "before evaluating."
        )
    thresholds = {
        "min_confidence": 0.0,
        "min_margin": 0.0,
        "min_module_score": _DEFAULT_MIN_MODULE_SCORE,
    }

    bulk = load_intent_eval_queries(eval_queries_path)
    bulk_records, bulk_modules, bulk_vectors = _predict(
        index, bulk, thresholds, model_name=model_name
    )
    _raise_on_leakage(index, bulk, bulk_vectors)

    split = split_eval_queries(bulk, slice_size=slice_size, seed=seed)
    by_id = {
        record.example_id: (record, vector, modules)
        for record, vector, modules in zip(bulk_records, bulk_vectors, bulk_modules)
    }
    tuning_records = tuple(by_id[q.id][0] for q in split.tuning)
    tuning_vectors = np.stack([by_id[q.id][1] for q in split.tuning])
    test_records = tuple(by_id[q.id][0] for q in split.test)
    test_vectors = np.stack([by_id[q.id][1] for q in split.test])
    test_modules = tuple(by_id[q.id][2] for q in split.test)

    legacy_pairs = [
        (record, vector)
        for record, vector in zip(tuning_records, tuning_vectors)
        if record.example_id.startswith(LEGACY_PREFIX)
    ]
    legacy_records = tuple(record for record, _ in legacy_pairs)
    legacy_vectors = (
        np.stack([vector for _, vector in legacy_pairs])
        if legacy_pairs
        else np.empty((0, tuning_vectors.shape[1]))
    )

    report: dict[str, Any] = {
        "index": {
            "size": index.size,
            "encoder": index.encoder,
            "fingerprint": index.fingerprint,
            "canonical": str(canonical_path),
            "low_support_modules": list(index.low_support_modules()),
        },
        "split": {
            "seed": seed,
            "slice_size": slice_size,
            "tuning_size": len(split.tuning),
            "test_size": len(split.test),
        },
        "bulk": _argmax_report(bulk_records),
        "modules": module_metrics_report(bulk_modules),
        "test_modules": module_metrics_report(test_modules),
        "leave_one_out": leave_one_out_route_accuracy(index),
    }

    hard_records = hard_vectors = None
    if hard_queries_path is not None:
        hard = load_intent_eval_queries(hard_queries_path)
        hard_records, hard_modules, hard_vectors = _predict(
            index, hard, thresholds, model_name=model_name
        )
        _raise_on_leakage(index, hard, hard_vectors)
        report["hard_modules"] = module_metrics_report(hard_modules)

    selected: dict[str, Any] | None = None
    probe_vectors: np.ndarray | None = None
    probes: tuple[tuple[str, str], ...] = ()
    if out_of_scope_path is not None:
        probes = load_out_of_scope_probes(out_of_scope_path)
        probe_vectors = encode_texts(
            [text for _, text in probes], model_name=model_name
        )
        report["threshold_tuning"] = _select_thresholds(
            index, tuning_records, tuning_vectors, probe_vectors
        )
        selected = report["threshold_tuning"]["selected"]

    # Every headline number below is reported at these hyperparameters -- the
    # winner of the tuning-only sweep above, or the unswept defaults when
    # there were no out-of-scope probes to tune against.
    top_k = selected["top_k"] if selected else TOP_K
    min_confidence = selected["min_confidence"] if selected else 0.0
    min_margin = selected["min_margin"] if selected else 0.0

    report["legacy_30"] = {
        **_serving_report(
            index,
            legacy_records,
            legacy_vectors,
            top_k=top_k,
            min_confidence=min_confidence,
            min_margin=min_margin,
        ),
        "tuned_on": True,
    }
    report["tuning"] = {
        **_serving_report(
            index,
            tuning_records,
            tuning_vectors,
            top_k=top_k,
            min_confidence=min_confidence,
            min_margin=min_margin,
        ),
        "tuned_on": True,
    }
    report["test_slice"] = {
        **_serving_report(
            index,
            test_records,
            test_vectors,
            top_k=top_k,
            min_confidence=min_confidence,
            min_margin=min_margin,
        ),
        "tuned_on": False,
    }
    if hard_records is not None:
        report["hard_40"] = {
            **_serving_report(
                index,
                hard_records,
                hard_vectors,
                top_k=top_k,
                min_confidence=min_confidence,
                min_margin=min_margin,
            ),
            "tuned_on": False,
        }

    if probe_vectors is not None:
        # Raw (ungated) confidences at the selected top_k -- separability
        # measures how well the two distributions separate, not what a
        # threshold happens to catch.
        test_decisions = _decide_batch(
            index, test_vectors, top_k=top_k, min_confidence=0.0, min_margin=0.0
        )
        probe_decisions = _decide_batch(
            index, probe_vectors, top_k=top_k, min_confidence=0.0, min_margin=0.0
        )
        out_of_scope = {
            "probes": len(probes),
            **separability_report(
                in_scope=[d.confidence for d in test_decisions],
                out_of_scope=[d.confidence for d in probe_decisions],
            ),
        }
        # Alias retained for the raw-cosine-units bar pinned in
        # test_intent_index_eval.py, which reads this key.
        out_of_scope["separation_margin"] = out_of_scope["raw_margin"]
        report["out_of_scope"] = out_of_scope

    report["top_k_sweep"] = {
        "note": (
            "Report-only. TOP_K stays 3 in serving (unchanged by this sweep): "
            "the accuracy/abstention trade it exposes -- test slice accuracy "
            "rises while out-of-scope separation falls as k grows -- should "
            "be decided once, together with a stronger encoder, not twice. "
            "See docs/training-and-evaluation.md."
        ),
        "rows": _sweep_top_k(
            index,
            test_records,
            test_vectors,
            hard_records,
            hard_vectors,
            probe_vectors,
        ),
    }

    report["headline"] = {
        "test_slice_accuracy": report["test_slice"]["accuracy"],
        "test_slice_size": report["split"]["test_size"],
        "hard_accuracy": report.get("hard_40", {}).get("accuracy"),
        "out_of_scope_auc": report.get("out_of_scope", {}).get("auc"),
        "cohens_d": report.get("out_of_scope", {}).get("cohens_d"),
        "leave_one_out_accuracy": report["leave_one_out"]["accuracy"],
        "selected_top_k": selected["top_k"] if selected else None,
        "selected_min_confidence": selected["min_confidence"] if selected else None,
        "selected_min_margin": selected["min_margin"] if selected else None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
