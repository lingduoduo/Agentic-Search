# Make the eval CI gates honest about when they're inactive

**Date:** 2026-07-19
**Status:** Approved

## Problem

`.github/workflows/eval-gate.yml` has two jobs that report green while gating
nothing:

1. **retrieval-eval-gate** — every step is guarded by
   `if: steps.check.outputs.exists == 'true'`, keyed on
   `data/eval/baseline_metrics.json`. That file is **gitignored and not
   committed** (only `data/eval/routing_labels.jsonl` is tracked under
   `data/eval/`), so a fresh CI checkout doesn't have it and the whole job
   silently skips green. Even if it were committed, the local copy is a zero
   stub (`{recall@10:0, ndcg@10:0, mrr:0}`) — a drop is `baseline - current`, so
   a zero baseline can never trip the gate.
2. **ragas-eval-gate** — every step is guarded by
   `if: steps.check.outputs.exists == 'true'`, keyed on
   `data/eval/ragas_baseline.json`, which **does not exist** in the repo. So the
   whole job silently skips and the check is green, having run nothing — with no
   log indication.

A real baseline can't be committed here (it needs a live retrieval stack + the
matching corpus/index), so the fix is honesty, not fabricated numbers: make CI
visibly state when a gate is inactive instead of showing a misleading green.

## Design

Scope: `.github/workflows/eval-gate.yml` + the baseline note + a docs pointer.
No fabricated baseline numbers, no stack runs, no change to the eval code.

1. **retrieval-eval-gate — detect the placeholder baseline.** In the
   "Check for regressions" step, if every gated metric in the baseline is `0`
   (the placeholder), emit a GitHub `::warning::` annotation and a
   `$GITHUB_STEP_SUMMARY` line: "Retrieval eval gate INACTIVE — baseline is a
   zero placeholder; not enforcing regressions." Keep the real comparison for
   when a genuine (non-zero) baseline is committed. Non-failing.

2. **ragas-eval-gate — make the missing-baseline skip visible.** Replace the
   silent whole-job skip with an always-running step: when
   `ragas_baseline.json` is absent, emit a `::warning::` + summary line "RAGAS
   eval gate INACTIVE — no baseline committed (see docs/training-and-evaluation.md)."
   The eval/compare steps still run only when the baseline exists. Non-failing,
   but the inactive state is now unmistakable in the run.

3. **Baseline note + docs.** Update `data/eval/baseline_metrics.json`'s `_note`
   to say it is a non-enforcing placeholder and point to the activation steps.
   Add a short "Activating the eval gates" note to
   `docs/training-and-evaluation.md` describing how to generate + commit real
   baselines (`baseline_metrics.json`, `ragas_baseline.json`) by running the eval
   harnesses against the canonical stack.

## Non-goals

- Not generating real baseline numbers (needs infra/stack decision).
- Not making the gates hard-fail on a missing/placeholder baseline (that would
  red-CI every PR until baselines are committed).
- No change to `eval_runner.py` / `ragas_eval.py` / `beir_eval.py`.

## Verification

- `python -c "import yaml, sys; yaml.safe_load(open('.github/workflows/eval-gate.yml'))"`
  parses cleanly.
- The inline Python placeholder-detection block runs against the committed
  zero-stub baseline and prints the INACTIVE warning (spot-checked locally).
- The RAGAS "inactive" notice step has no `if:` guard, so it runs on a missing
  baseline.
