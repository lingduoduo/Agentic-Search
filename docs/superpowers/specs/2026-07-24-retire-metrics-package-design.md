# Retire the dead metrics package

**Date:** 2026-07-24
**Branch:** `chore/retire-metrics-package` (off `chore/retire-nlp-package`; PR opens against `main` after #468 merges)
**Status:** design approved (scope), pending spec review

## Context

Follow-up to the NLP-package retirement (#468). An audit of `src/internal/metrics/`
found the **entire package — 2,391 LOC across 17 modules — is dead, unwired Onyx
observability heritage**, with **zero external importers** anywhere (src, tests,
examples, docker), no `python -m` entrypoint, no startup wiring, and no tests. The
web app performs no Prometheus instrumentation at all.

Two provenance groups, both dead:
- **Campaign-orphaned** (monitored machinery deleted in #461–#464 / #468):
  `celery_task_metrics`, `indexing_task_metrics`, `indexing_pipeline`,
  `indexing_pipeline_setup`, `connector_health_metrics`, `perm_sync_metrics`,
  `pruning_metrics`, `deletion_metrics`, `embedding`, `image_processing`.
- **Generic HTTP-metrics scaffolding, never wired** into the app:
  `metrics_server`, `prometheus_setup`, `metrics_auth`, `per_tenant`,
  `slow_requests`, `postgres_connection_pool`.

The modules import each other (self-contained cluster) but nothing external imports
any of them. `prometheus-client` is used **only** by this package.

## Goal

Delete the entire `src/internal/metrics/` package and drop the now-unused
`prometheus-client` dependency, with zero live-behavior change.

## Sequencing (dependency on #468)

`metrics/embedding.py` is imported by `natural_language_processing/search_nlp_models.py`,
which #468 deletes. On `main` today that consumer still exists, so the *whole-package*
deletion is only clean once #468 has merged. This branch is cut from
`chore/retire-nlp-package` (where the NLP package is already gone) so the package is
genuinely orphaned in the working state; the PR is opened against `main` **after #468
merges** (rebased onto main — the file sets don't overlap). Verified: no metrics
module imports `metrics.embedding`, so the package deletes cleanly as a unit.

## Scope

### Delete
- `src/internal/metrics/` — the whole directory (all 17 modules + `__init__.py`).

### Edit
- `requirements.txt` — remove `prometheus-client>=0.20.0`.
- `requirements-unit-test.txt` — remove `prometheus-client>=0.20.0`.

### Out of scope
- The separate `document_index/utils.setup_logger` relocation (the NLP-PR follow-up).
- Any change to the web app's observability (it has none via this package; unchanged).

## Verification / success criteria

1. `grep -rn "internal.metrics" src/ tests/ examples/ docker/` returns nothing (no
   importer survives — already true today; re-confirm after deletion).
2. `grep -rn "prometheus_client\|prometheus_fastapi_instrumentator" src/` returns
   nothing (the only user was the deleted package) — so the dep drop is safe.
3. `python -c "import src"` succeeds; `ruff check .` + `ruff format --check .` pass;
   `pytest` green (no test referenced the package).
4. `src/__init__.py` does not reference `internal.metrics` (no re-export trap —
   confirmed; the `PerformanceMetrics` symbol there is unrelated, from `agents.core.state`).

## Risks

Low. The whole package is verified dead (zero external references, no entrypoint, no
tests). Dropping `prometheus-client` from the manifests does not affect the already-
installed test env; the only risk would be a surviving `prometheus_client` import,
excluded by criterion 2. The only real subtlety is the #468 sequencing, handled by
branching from the NLP branch and merging after #468.
