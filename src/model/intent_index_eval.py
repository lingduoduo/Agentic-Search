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
from .intent_eval_split import (
    DEFAULT_SEED,
    DEFAULT_SLICE_SIZE,
    LEGACY_PREFIX,
    split_eval_queries,
)
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
# untouched test set. Imported from intent_eval_split, the module that owns
# the split, so the two definitions cannot drift apart.

# The tuning-slice threshold sweep, fixed by the spec: never touch the test
# slice or hard slice while choosing serving hyperparameters.
_SWEEP_MIN_CONFIDENCES = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
# Margins span two encoders' scales. The four large values are MiniLM's; the
# low end is derived from the *tuning* slice's own margin quantiles under
# e5-small-v2 (min 0.0008, p25 0.0116, median 0.0188, p75 0.0280, max 0.0676),
# because e5 compresses every route score into a narrow band and a grid that
# starts at 0.02 abstains on more than half of everything before it begins.
# Derive any future extension from the tuning slice too -- never from the test
# slice, whose quantiles are a different (and off-limits) distribution.
_SWEEP_MIN_MARGINS = (0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.05, 0.08, 0.12)
_MIN_COVERAGE = 0.60
# Report-only, for _sweep_top_k's table. `top_k` is NOT swept for selection --
# see _select_thresholds.
_SWEEP_TOP_K = (3, 5, 8, 15, 25)

# Mirrors AppSettings.intent_min_module_score's default (src/internal/configs/
# app_configs.py) so evaluation scores modules with the same bar serving uses.
_DEFAULT_MIN_MODULE_SCORE = 0.84

# Module-score grid, derived from the *tuning* slice's own quantiles under
# e5-small-v2 (over the winning route's candidate modules, the only ones
# _emit_modules considers): p0 0.7450, p25 0.8047, p50 0.8221, p75 0.8374,
# p100 0.8941. The legacy 0.4500 is kept as the first row so the sweep records
# the status quo it replaces -- it is below every score e5 produces, so it
# emits every well-supported module of the route and is what drove module
# precision to ~0.2 and joint accuracy to 0.0.
#
# Derive any future grid from the tuning slice too, never the test slice. A
# grid in the previous encoder's units selects nothing at all: that is exactly
# what happened to the margin grid in #512.
_SWEEP_MIN_MODULE_SCORES = (
    0.45,
    0.76,
    0.78,
    0.80,
    0.82,
    0.84,
    0.86,
    0.88,
    0.90,
)


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
    min_module_score: float = _DEFAULT_MIN_MODULE_SCORE,
) -> list[KnnDecision]:
    thresholds = {
        "min_confidence": min_confidence,
        "min_margin": min_margin,
        "min_module_score": min_module_score,
        "top_k": top_k,
    }
    return [index.decide(vector, **thresholds) for vector in vectors]


def _decide_records(
    index: IntentIndex,
    queries,
    vectors: np.ndarray,
    *,
    top_k: int,
    min_module_score: float = _DEFAULT_MIN_MODULE_SCORE,
) -> tuple[tuple[IntentPredictionRecord, ...], tuple[ModulePredictionRecord, ...]]:
    """Route + module predictions at a specific top_k, from already-encoded vectors.

    Argmax only (min_confidence=min_margin=0.0), matching ``_argmax_report``'s
    semantics. Rebuilds the diagnostic prediction/module records at the
    selected top_k without re-encoding: the first, fixed-top_k decide pass in
    ``_predict`` only exists to get vectors (for the split, for leakage
    detection) before ``top_k`` is known; its own ``.predicted``/``.modules``
    would otherwise silently stay at ``TOP_K`` regardless of what got
    selected, which is inconsistent with every other block in the report.
    Latency is not recomputed here -- it was already captured by that first
    pass and nothing downstream reads it.
    """
    decisions = _decide_batch(
        index,
        vectors,
        top_k=top_k,
        min_confidence=0.0,
        min_margin=0.0,
        min_module_score=min_module_score,
    )
    records = tuple(
        IntentPredictionRecord(
            example_id=query.id,
            expected=query.label,
            predicted=decision.route,
            confidence=decision.confidence,
            latency_ms=0.0,
            mechanism="model",
        )
        for query, decision in zip(queries, decisions)
    )
    modules = tuple(
        ModulePredictionRecord(
            example_id=query.id,
            expected=tuple(query.modules),
            predicted=tuple(decision.modules),
            route_correct=decision.route == query.label,
        )
        for query, decision in zip(queries, decisions)
    )
    return records, modules


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
    """Sweep (min_confidence, min_margin) on the tuning slice only, at fixed k.

    Tuned exclusively against the tuning split and the out-of-scope probes --
    the test slice and hard slice are never consulted. Selects the
    combination with the highest served accuracy at coverage >= 0.60,
    breaking ties toward higher out-of-scope deferral: the rule fixed in
    advance by the spec. ``leave_one_out_route_accuracy`` is not part of this
    key -- see its docstring for why it would be a biased selector.

    ``top_k`` is **pinned at the shipped ``TOP_K``** rather than swept, and
    that is a safety property, not a convenience. The reported headline is
    argmax route accuracy, which is abstention-blind: it depends on ``top_k``
    and on nothing else this function chooses. Sweeping ``top_k`` therefore
    couples a tuning-slice search to the held-out headline, so widening the
    grid after seeing the headline could move it. Pinning ``k`` severs that
    coupling arithmetically -- with ``k`` fixed, no choice made here can
    change ``test_slice.accuracy`` by any amount -- which is what makes the
    threshold grid free to be re-derived whenever the encoder changes.
    ``_sweep_top_k`` still reports the per-k table, on the tuning slice, as
    evidence for a separate and deliberate decision.
    """
    sweep: list[dict[str, Any]] = []
    top_k = TOP_K
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


def _select_module_threshold(
    index: IntentIndex,
    tuning_queries,
    tuning_vectors: np.ndarray,
    *,
    top_k: int,
) -> dict[str, Any]:
    """Sweep ``min_module_score`` on the tuning slice only.

    The rule is fixed in advance (see
    docs/superpowers/plans/2026-08-14-intent-module-threshold.md): **highest
    module macro-F1, ties broken toward the lower threshold.** Macro-F1 rather
    than precision-at-a-recall-floor because module emission is multi-label and
    a recall floor would be a constant invented after seeing the curve. Ties to
    the lower threshold because over-emitting is visible in precision while
    under-emitting is hidden by ``_emit_modules``'s top-1 fallback.

    ``joint_accuracy`` is recorded for every row but never selects: it is an
    exact-set match over 70 queries and would chase the support distribution.

    This sweep cannot move the route. ``_emit_modules`` runs *after*
    ``decide()`` has already chosen from ``route_scores``, so no value here
    changes ``test_slice.accuracy`` -- an invariant asserted by
    ``test_the_module_sweep_never_changes_the_route``.
    """
    sweep: list[dict[str, Any]] = []
    for min_module_score in _SWEEP_MIN_MODULE_SCORES:
        _, modules = _decide_records(
            index,
            tuning_queries,
            tuning_vectors,
            top_k=top_k,
            min_module_score=min_module_score,
        )
        metrics = module_metrics_report(modules)
        emitted = [len(record.predicted) for record in modules]
        sweep.append(
            {
                "min_module_score": min_module_score,
                "macro_f1": metrics["macro_f1"],
                "joint_accuracy": metrics["joint_accuracy"],
                "mean_modules_emitted": sum(emitted) / len(emitted) if emitted else 0.0,
            }
        )

    ranked = sorted(sweep, key=lambda row: (-row["macro_f1"], row["min_module_score"]))
    return {"sweep": sweep, "selected": ranked[0] if ranked else None, "tuned_on": True}


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
    in_scope_vectors: np.ndarray,
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
    in_scope = [index.decide(v, **kwargs).confidence for v in in_scope_vectors]
    out_of_scope = [index.decide(v, **kwargs).confidence for v in probe_vectors]
    return sum(in_scope) / len(in_scope) - sum(out_of_scope) / len(out_of_scope)


def _sweep_top_k(
    index: IntentIndex,
    tuning: tuple[IntentPredictionRecord, ...],
    tuning_vectors: np.ndarray,
    probe_vectors: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Report-only sweep of TOP_K over the already-built index and encoder.

    Evidence for a later decision, not a selection: TOP_K stays 3 in serving
    regardless of this table. Draws only on the tuning slice and the
    out-of-scope probes -- never the test slice, and never hard-40, which is
    held-out test data by this project's own split (see
    docs/training-and-evaluation.md) even though it is not drawn from the
    same clean-query pool. This is the fitting curve for the one
    hyperparameter this task moved into an automated sweep, and publishing
    that curve computed on data no hyperparameter is allowed to see would
    hand test-set fitting straight back to the human reading the report,
    even though nothing in code selects from it. See the module-level
    ``_SWEEP_TOP_K`` comment and docs/training-and-evaluation.md.
    """
    rows: list[dict[str, Any]] = []
    for top_k in _SWEEP_TOP_K:
        row: dict[str, Any] = {
            "top_k": top_k,
            "tuning_accuracy": _accuracy_at_top_k(index, tuning, tuning_vectors, top_k),
            "leave_one_out_accuracy": leave_one_out_route_accuracy(index, top_k=top_k)[
                "accuracy"
            ],
        }
        if probe_vectors is not None and len(probe_vectors):
            row["separation_margin"] = _separation_margin_at_top_k(
                index, tuning_vectors, probe_vectors, top_k
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
    bulk_records, _, bulk_vectors = _predict(
        index, bulk, thresholds, model_name=model_name
    )
    _raise_on_leakage(index, bulk, bulk_vectors)

    split = split_eval_queries(bulk, slice_size=slice_size, seed=seed)
    # Only .expected (via the record) and the vector are needed downstream --
    # .predicted/.modules on this first, fixed-TOP_K pass get rebuilt at the
    # selected top_k below, once it is known.
    by_id = {
        record.example_id: (record, vector)
        for record, vector in zip(bulk_records, bulk_vectors)
    }
    tuning_records = tuple(by_id[q.id][0] for q in split.tuning)
    tuning_vectors = np.stack([by_id[q.id][1] for q in split.tuning])
    test_records = tuple(by_id[q.id][0] for q in split.test)
    test_vectors = np.stack([by_id[q.id][1] for q in split.test])

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
        "leave_one_out": leave_one_out_route_accuracy(index),
    }

    hard = hard_records = hard_vectors = None
    if hard_queries_path is not None:
        hard = load_intent_eval_queries(hard_queries_path)
        hard_records, _, hard_vectors = _predict(
            index, hard, thresholds, model_name=model_name
        )
        _raise_on_leakage(index, hard, hard_vectors)

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

    # bulk/modules/test_modules/hard_modules were only ever decided at the
    # first pass's fixed TOP_K (needed before top_k was known, to get vectors
    # for the split and the sweep). Rebuild them now at the selected top_k so
    # every block in the report reflects the same hyperparameters.
    # Chosen on the tuning slice only, before any module block below is built,
    # so every reported module number reflects the threshold serving will use.
    report["module_threshold_tuning"] = _select_module_threshold(
        index, split.tuning, tuning_vectors, top_k=top_k
    )
    module_selected = report["module_threshold_tuning"]["selected"]
    min_module_score = (
        module_selected["min_module_score"]
        if module_selected
        else _DEFAULT_MIN_MODULE_SCORE
    )

    bulk_records, bulk_modules = _decide_records(
        index, bulk, bulk_vectors, top_k=top_k, min_module_score=min_module_score
    )
    _, test_modules = _decide_records(
        index,
        split.test,
        test_vectors,
        top_k=top_k,
        min_module_score=min_module_score,
    )
    report["bulk"] = _argmax_report(bulk_records)
    report["modules"] = module_metrics_report(bulk_modules)
    report["test_modules"] = module_metrics_report(test_modules)
    if hard_records is not None:
        hard_records, hard_modules = _decide_records(
            index,
            hard,
            hard_vectors,
            top_k=top_k,
            min_module_score=min_module_score,
        )
        report["hard_modules"] = module_metrics_report(hard_modules)

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
        report["out_of_scope"] = out_of_scope

    report["top_k_sweep"] = {
        "note": (
            "Report-only, computed on the tuning slice (never test): TOP_K "
            "stays 3 in serving (unchanged by this sweep). The "
            "accuracy/abstention trade it exposes -- tuning accuracy rises "
            "while out-of-scope separation falls as k grows -- should be "
            "decided once, together with a stronger encoder, not twice. See "
            "docs/training-and-evaluation.md."
        ),
        "rows": _sweep_top_k(
            index,
            tuning_records,
            tuning_vectors,
            probe_vectors,
        ),
    }

    report["headline"] = {
        # Argmax (abstention-blind) accuracy on the untouched test slice --
        # this is the honest headline number.
        "test_slice_accuracy": report["test_slice"]["accuracy"],
        "test_slice_size": report["split"]["test_size"],
        "hard_accuracy": report.get("hard_40", {}).get("accuracy"),
        "out_of_scope_auc": report.get("out_of_scope", {}).get("auc"),
        "cohens_d": report.get("out_of_scope", {}).get("cohens_d"),
        "leave_one_out_accuracy": report["leave_one_out"]["accuracy"],
        "selected_top_k": selected["top_k"] if selected else None,
        "selected_min_confidence": selected["min_confidence"] if selected else None,
        "selected_min_margin": selected["min_margin"] if selected else None,
        # The metric the sweep actually maximized -- served (abstention-
        # gated) accuracy on the *tuning* slice at the selected
        # hyperparameters. Different quantity, different slice from
        # test_slice_accuracy above; do not compare them as if they were the
        # same number measured twice.
        "tuning_served_accuracy": selected["served_accuracy"] if selected else None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
