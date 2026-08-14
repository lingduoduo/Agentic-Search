# Plan: re-derive `min_module_score` in e5 units

Spec: `docs/superpowers/specs/2026-08-14-intent-min-module-score-design.md`

## Pre-registered selection rule

**Committed before the sweep is run, and not to be changed after seeing its output.**

> Select the `min_module_score` that **maximizes module macro-F1 on the tuning
> slice**. Break ties toward the **lower** threshold.

Two choices inside that, both made now for stated reasons:

- **Macro-F1, not precision-at-a-recall-floor.** Module emission is multi-label,
  macro-F1 is the standard multi-label summary, and it balances the two failure
  directions without my having to invent a recall floor — an arbitrary constant
  chosen after seeing the curve is exactly the fitting this repo has already paid
  for twice.
- **Ties toward the lower threshold.** A too-low threshold over-emits, which is
  visible in precision. A too-high threshold silently drops a module a query
  genuinely had, and `_emit_modules`'s top-1 fallback hides it. The recoverable
  failure wins ties.

**Joint accuracy is a reported result, never the selector.** It is an exact-set
match and would push the grid toward whichever extreme happens to suit the
support distribution of 111 queries.

## The mechanism this sweep actually moves

`_emit_modules` emits every well-supported module of the winning route scoring
`>= min_module_score`, and falls back to the single best when none clear it. So
the threshold interpolates between two regimes:

- **low** → every module of the route (today: recall ≈ 1.0, precision ≈ 0.2)
- **high** → the top-1 fallback for every query (precision up, recall down on
  genuinely multi-module queries)

The useful setting is wherever between those the tuning slice says it is. Note
that a high threshold does **not** produce empty module sets — the fallback
guarantees at least one — so "no modules emitted" is not a failure mode to guard.

## Steps

### 1. Derive the grid from the tuning slice's own module-score quantiles

Per the spec, and for the same reason the margin grid was re-derived in #512: the
`0.45` start is in MiniLM units and would select nothing.

→ verify: the grid spans the tuning slice's observed module-score range, so at
least one endpoint changes the emitted set relative to the other.

### 2. Sweep, selecting by the rule above

Add `_select_module_threshold`, mirroring `_select_thresholds`: tuning slice only,
`top_k` pinned at `TOP_K`, test and hard slices never consulted.

→ verify: `evaluation_report.json` gains a `module_threshold_tuning` block with
the full sweep and the selection, marked `tuned_on: true`.

### 3. Re-measure on the test slice

→ verify against the spec's acceptance criteria:
- module joint accuracy on the test slice `> 0.0` (it is `0.0` today)
- module precision materially above `≈0.2` without recall collapsing
- **test-slice route accuracy still exactly `0.7928`** — this is the invariant. A
  moved headline means module emission leaked into the routing path, which would
  be a real defect, not a tuning result.

### 4. Ship the default

Update `intent_min_module_score` in `src/internal/configs/app_configs.py` and
`_DEFAULT_MIN_MODULE_SCORE` in `src/model/intent_index_eval.py` together — they
are documented as mirroring each other and a test pins that.

→ verify: `pytest tests/unit/test_intent_index_eval.py tests/unit/test_intent_canonical_data.py`
plus the config and routing unit tests.

### 5. Update the docs

`docs/training-and-evaluation.md`: the units-trap table and the known-limitation
bullet both currently say this threshold is un-derived. Both change, along with
the module macro-F1 / joint-accuracy row in the headline table.

→ verify: no surviving claim that `min_module_score` is un-derived or dead.

## Risk

Low. Modules are diagnostics and cannot change a route, so the blast radius of a
wrong value is a worse diagnostic, not a worse answer. The invariant in step 3 is
what proves that claim rather than assuming it.
