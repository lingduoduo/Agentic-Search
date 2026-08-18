# Plan — consolidate the intent modules into `src/model/intent/`

Design: [`2026-08-18-intent-package-consolidation-design.md`](../specs/2026-08-18-intent-package-consolidation-design.md)

## 1. Establish what is live before deleting anything

- Trace every consumer of the nine `intent_*` modules: `ml_intent.py`,
  `intent_routing.py`, `examples/run_agentic_search.py`,
  `examples/measure_intent_operating_point.py`, `src/__init__.py`'s lazy export
  table, `src/model/__init__.py`, and the CI eval gate.
- For each exported symbol, distinguish *called* from *merely named*. Two traps
  here: `evaluate_intent_predictions` appears in `intent_index_eval.py` only
  inside a docstring, and `modules`/`composite` look unused but are deliberately
  persisted to telemetry by `intent_routing.py`.
- Verify the #530 promotion checklist is prose, not a `compare_for_promotion`
  call site.
- **Verify:** every symbol classified live or dead with a named call site or a
  demonstrated absence of one.

## 2. Move the files with history preserved

- `git mv` each of the nine modules into `src/model/intent/`, then concatenate
  the three groups that merge (`model`, `evaluation`, `cli`), stripping
  duplicated `__future__` and import blocks and merging the docstrings so each
  merged section keeps the rationale it carried.
- Re-home `build_index`/`fingerprint` into `data.py` and
  `check_leakage`/`LEAKAGE_COSINE` into `evaluation.py`.
- **Verify:** `ast.parse` each new module; `git status` shows renames, not
  add/delete pairs.

## 3. Cut the retired classifier's machinery from `metrics.py`

- Delete the ten public symbols and four private helpers listed in the design.
- Drop the now-unused `confusion_matrix` sklearn import and `Mapping` typing
  import.
- Rewrite the module docstring to record what was removed and why, so the next
  reader does not have to re-derive it from git history.
- **Verify:** `metrics.py` is 227 LOC; `ruff check` passes; every surviving
  function still has a live caller.

## 4. Write `__init__.py` and repoint every consumer

- Re-export only `model` and `data`; leave `metrics`/`evaluation` behind module
  paths so scikit-learn stays off the request path.
- Update `src/__init__.py`'s lazy table, `src/model/__init__.py`,
  `ml_intent.py`, both examples, the CI eval gate, `docs/configuration.md`, and
  `docs/training-and-evaluation.md`. Dated plan/spec/context-pack documents keep
  the old paths — they are records of past work, not runnable references.
- Point monkeypatched consumers at `src.model.intent.model`, not the package.
- **Verify:** `grep -rn "src\.model\.intent_"` returns nothing outside
  `docs/superpowers/`; the CLI's three commands run.

## 5. Repoint and prune the tests

- Rewrite imports across all 13 `test_intent_*` files plus `test_ml_intent.py`,
  `test_run_agentic_search.py`, and `tests/unit/servers/web/test_web_experience_app.py`.
- Delete the 15 tests covering deleted functions.
- Rename test files to mirror the modules: `test_intent_evaluation.py` →
  `test_intent_metrics.py`, `test_intent_index_eval.py` →
  `test_intent_evaluation.py`, `test_intent_index_cli.py` → `test_intent_cli.py`,
  `test_intent_knn.py` → `test_intent_model.py`.
- **Verify:** collected-test delta against `origin/main` is exactly 15.

## 6. Verify against a real baseline

- Run the full unit suite on this branch and on `origin/main` **in a worktree
  with `data/` symlinked in** — `data/` is gitignored, so a bare worktree skips
  the tests that read the built index and silently misreports the baseline.
- **Verify:** identical failure set on both sides. Expected: 10 failures, all
  from the absent HuggingFace download of `intfloat/e5-small-v2`. Any failure
  outside that set is a regression and must be fixed, not explained.
