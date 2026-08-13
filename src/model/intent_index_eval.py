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

from .intent_data import load_intent_eval_queries, load_out_of_scope_probes
from .intent_encoder import encode_texts
from .intent_evaluation import (
    IntentPredictionRecord,
    ModulePredictionRecord,
    module_metrics_report,
    realistic_accuracy_report,
)
from .intent_index_cli import check_leakage
from .intent_knn import INDEX_FILENAME, IntentIndex

# Legacy ids are `eval-<route>-NN`; queries added later use `bulk-NNN`. The
# legacy 30 were iterated against during canonical-set curation, so they are
# contaminated; the bulk-prefixed queries are the clean, honest slice.
LEGACY_PREFIX = "eval-"


def _predict(index, queries, thresholds):
    records, modules, latencies = [], [], []
    texts = [query.text for query in queries]
    vectors = encode_texts(texts)
    for query, vector in zip(queries, vectors):
        start = perf_counter()
        decision = index.decide(vector, **thresholds)
        latencies.append((perf_counter() - start) * 1_000)
        records.append(
            IntentPredictionRecord(
                example_id=query.id,
                expected=query.label,
                predicted=decision.route,
                confidence=decision.confidence,
                latency_ms=latencies[-1],
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
    return records, modules, vectors


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

    leaks = check_leakage(index, [q.text for q in bulk], bulk_vectors)
    if leaks:
        raise ValueError(
            f"{len(leaks)} evaluation queries leak into the canonical set; "
            f"first: {leaks[0]}"
        )

    legacy = tuple(r for r in bulk_records if r.example_id.startswith(LEGACY_PREFIX))
    # The clean slice: bulk-prefixed queries the canonical set was never
    # iterated against. This is the honest accuracy number.
    clean = tuple(r for r in bulk_records if not r.example_id.startswith(LEGACY_PREFIX))
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
        "bulk": realistic_accuracy_report(bulk_records, threshold=0.0),
        "legacy_30": realistic_accuracy_report(legacy, threshold=0.0),
        "clean_151": realistic_accuracy_report(clean, threshold=0.0),
        "modules": module_metrics_report(bulk_modules),
        "clean_modules": module_metrics_report(clean_modules),
    }

    if hard_queries_path is not None:
        hard = load_intent_eval_queries(hard_queries_path)
        hard_records, hard_modules, _ = _predict(index, hard, thresholds)
        report["hard"] = realistic_accuracy_report(hard_records, threshold=0.0)
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

    report["headline"] = {
        "bulk_accuracy": report["bulk"]["accuracy"],
        "legacy_30_accuracy": report["legacy_30"]["accuracy"],
        "clean_151_accuracy": report["clean_151"]["accuracy"],
        "hard_accuracy": report.get("hard", {}).get("accuracy"),
        "module_macro_f1": report["modules"]["macro_f1"],
        "joint_accuracy": report["modules"]["joint_accuracy"],
        "separation_margin": report.get("out_of_scope", {}).get("separation_margin"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
