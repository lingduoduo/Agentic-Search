# Consolidate `src/model/intent_*.py` into one package, and drop the retired classifier's machinery

## Problem

`src/model/` held nine top-level `intent_*.py` modules totalling ~2,740 LOC.
Two separate problems had accumulated:

**1. The layout no longer said what belonged together.** Nine sibling files at
the top of `src/model/` — next to `generation.py`, `serving.py`, and
`tensor_helper.py`, which have nothing to do with intent — gave no signal about
which three modules are on the request path and which six are offline tooling.
`intent_index_eval.py` and `intent_index_cli.py` imported each other, with the
cycle broken by a function-local import.

**2. Half of `intent_evaluation.py` was machinery for a model that no longer
exists.** When routing moved from a trained MLP to the canonical-example index
(#511), that module was retained rather than deleted, on the stated theory that
its metrics are model-agnostic and would be reused. The index harness never
reused them: it reimplemented its own threshold sweep in `intent_index_eval.py`.
The following had no caller anywhere outside their own unit tests —

`evaluate_intent_predictions`, `IntentEvaluationReport`,
`select_confidence_threshold`, `calibration_report`,
`out_of_scope_abstention_rate`, `compose_candidate_cascade`,
`authoritative_routes_match`, `compare_for_promotion`, `PromotionCriteria`,
`PromotionDecision`

— along with the private helpers `_gate`, `_relative_gate`,
`_validated_confidences`, and `_percentile`.

The module docstring named this state explicitly and asked for a reviewed
decision rather than an inference from missing callers. This is that decision.

The promotion checklist pre-registered in #530 is **prose**, scored by hand
against the harness headline numbers. It never called `compare_for_promotion`,
so deleting the gate code does not remove the checklist.

## What this does not change

Behaviour. Serving, the index format, the CLI's three commands, the emitted
`evaluation_report.json`, and every threshold are untouched. The 14-module
multi-label taxonomy and the `composite` flag stay exactly as they are: nothing
acts on them, but `intent_routing.py` deliberately persists both to production
telemetry to gather data for a future plan-aware router, and that instrument is
still running.

## Design

Five modules under `src/model/intent/`, layered so each imports only from those
above it — a clean DAG with no function-local imports needed to break cycles:

| Module | Was | Contents |
| --- | --- | --- |
| `model.py` | `intent_taxonomy` + `intent_encoder` + `intent_knn` | Taxonomy, encoder, `IntentIndex`. The serving path. |
| `data.py` | `intent_data` + `build_index` | Validated loaders, and the build that turns them into an index. |
| `metrics.py` | `intent_evaluation` | Scoring over prediction records. |
| `evaluation.py` | `intent_eval_split` + `intent_index_eval` + `check_leakage` | The tuning/test split and the offline harness. |
| `cli.py` | `intent_index_cli` + `intent_seed` | `seed` / `build` / `evaluate` entry points. |

Three decisions worth recording:

**`build_index` moves to `data.py`.** It was in `cli.py`, which forced
`evaluation.py` to import from the CLI for `_fingerprint` and `check_leakage`,
which in turn forced `cli.py` to import `run_index_evaluation` inside a function
body. Homing the build with the data it loads, and the leakage check with the
evaluation that is its only caller, removes the cycle outright.

**`metrics` and `evaluation` are not re-exported from `__init__.py`.**
`metrics` imports scikit-learn; nothing on the request path should pay for that
import. They are reached by module path.

**Consumers that get monkeypatched import from the defining submodule, not the
package.** A package-level re-export binds its own reference at import time, so
patching `intent.model.encode_texts` does not reach an alias held by
`intent/__init__.py`. `ml_intent.py` and `run_agentic_search.py` therefore
import from `src.model.intent.model` directly, with a comment saying why.

## Result

- `src/model/intent/`: 2,364 LOC across 5 modules + `__init__.py`, down from
  2,740 across 9 files. `metrics.py` alone drops 705 → 227.
- 15 tests removed, all of them exercising deleted functions.
- Test files renamed to mirror the modules they cover.
- No behavioural change; no test regressions.
