# Evaluation Results panel in the Dev Console

**Date:** 2026-07-19
**Status:** Approved

## Problem

Training and evaluation are entirely offline (CLI). Eval runs leave result files
on disk (BEIR/RAGAS/`eval_runner` write flat `{metric: value}` JSON via
`--output`; Bamboogle writes per-example JSONL rows + a `BamboogleSummary`), but
there is no way to see any of it in the web UI. The Dashboard should surface the
latest evaluation results.

Scope decisions (agreed): **evaluation results only** (not training curves), in
the **Dev Console** (dev-only, `VITE_DEBUG_PANELS`), fed by a new `/api/debug/*`
endpoint that reads a configurable results directory. Read-only snapshots, no
charts.

## Design

### 1. Results directory convention

A configurable directory, `AGENTIC_SEARCH_EVAL_RESULTS_DIR`, default `data/eval/`.
Eval runs write their `--output` JSON there (`baseline_metrics.json` already
lives in `data/eval/`).

### 2. Backend — `GET /api/debug/eval-results`

Add to `src/internal/servers/web/debug_router.py`:
- Resolve the results dir from `AGENTIC_SEARCH_EVAL_RESULTS_DIR` (default
  `data/eval/`).
- List `*.json` files directly in that dir (non-recursive; `*.jsonl` per-example
  row files are excluded).
- Parse each: read JSON, keep only top-level keys whose value is an `int`/`float`
  (and not `bool`) as `metrics`; ignore `_note` and any non-numeric/nested keys.
- Return `{"results": [{"name": <filename>, "modified": <mtime epoch seconds>,
  "metrics": {k: number}}, ...]}` sorted by `modified` descending. Files with no
  numeric keys are still listed (empty `metrics`) so an all-zero placeholder is
  visible.
- Missing dir → `{"results": []}`. Read-only; confined to the configured dir;
  `*.json` only; no path traversal (list the dir, don't accept a client path).

### 3. Bamboogle participation

`run_bamboogle_eval` writes only JSONL rows today. Add a summary-JSON write: after
`evaluate_bamboogle` returns the `BamboogleSummary`, also write
`dataclasses.asdict(summary)` to `<output_stem>.summary.json` alongside the rows
(a flat `{num_examples, exact_match, contains_match, avg_reward}` file the panel
can read). If the output path is inside the results dir, it appears in the panel.

### 4. Frontend — `EvalResultsPanel`

New `web/src/components/debug/EvalResultsPanel.tsx`:
- Fetch `/api/debug/eval-results` on mount via a new `getEvalResults()` in
  `web/src/api.ts` (returns `EvalResultFile[]`, a new type in `web/src/types.ts`).
- Render one card per result file: filename, relative/ISO modified time, and a
  small two-column table of metric → formatted number (`toFixed(4)` for floats,
  integers as-is). A file with empty `metrics` shows "no numeric metrics".
- Empty state: "No eval results yet — run an eval with `--output` into
  `data/eval/` (see docs/training-and-evaluation.md)."
- Registered in `web/src/components/debug/DevConsole.tsx` alongside the existing
  panels.

## Scope / non-goals

- No training-metric curves (`metrics.jsonl`) — deliberate future follow-up.
- No charts — metric cards/tables only.
- No live streaming / websockets; read-on-load (a manual refresh button is fine,
  not required).
- No new prod surface: the panel is behind `VITE_DEBUG_PANELS`, the endpoint is
  under the existing `/api/debug` router.
- No parsing of per-example JSONL row files.

## Verification

- Backend: unit test for the endpoint — a tmp results dir with a flat metrics
  JSON, a `_note`-bearing zero stub, and a non-JSON file → returns the numeric
  metrics, ignores `_note`, sorts by mtime, and returns `[]` for a missing dir.
- Bamboogle: unit test that `run_bamboogle_eval`'s summary-write produces a
  `<stem>.summary.json` with the `BamboogleSummary` fields.
- Frontend: `EvalResultsPanel` test — renders metric rows from a mocked
  `getEvalResults()` and shows the empty state when the list is empty.
- `VITE_DEBUG_PANELS=1` Dev Console shows the panel; with `data/eval/` populated
  it lists the result files.
