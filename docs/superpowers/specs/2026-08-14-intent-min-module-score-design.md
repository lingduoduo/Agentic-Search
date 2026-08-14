# Re-derive `min_module_score` in the serving encoder's units

## Status

Deferred through #512–#518. Owed before any promotion decision. Not started.

## Problem

`AGENTIC_SEARCH_INTENT_MIN_MODULE_SCORE` defaults to `0.45`. It is a **cosine
similarity**, and it was derived when the router used `all-MiniLM-L6-v2`. The
encoder is now `intfloat/e5-small-v2`, which compresses cosines into a narrow
high band. The threshold was never re-derived.

Measured on the current index over the 111-query test slice — every module score
for every query, 1554 values:

```
min 0.7428   max 0.8943   values below the 0.45 threshold: 0 / 1554
```

The gate is not merely mistuned. **It is dead**: it cannot fire for any input.

## Consequence

`_emit_modules` returns "every well-supported module of the winning route,"
because the `score >= min_module_score` filter admits everything. From
`evaluation_report.json`:

| | value |
|---|---|
| module recall | ≈ 1.0 (many modules at exactly `1.0`) |
| module precision | ≈ 0.2 (`current_info` `0.194`, `compare` `0.111`) |
| module macro-F1, test slice | `0.3492` |
| **module joint accuracy** | **`0.0`** |

Joint accuracy is `0.0` on all three slices — not one of 181 queries gets its
full module set right. Under MiniLM it was `0.2318`.

## What is and is not at risk

**Routing is unaffected.** Modules are diagnostics; `IntentIndex.decide` picks
the route from `route_scores` alone and `_emit_modules` runs after the route is
already chosen. No request is routed differently because of this.

What is broken is the *diagnostic itself*. The module fields in
`evaluation_report.json` currently carry no information, so any future decision
that wants to read them — a plan-aware router, a module-conditioned prompt — has
no signal to read. This is a measurement outage, not a serving defect, which is
why it has been safe to defer and why it must not be deferred indefinitely.

## Approach

Re-derive on the **tuning slice**, never the test slice, exactly as
`min_margin` was re-derived in #512:

1. Compute the module-score quantiles on the tuning slice under e5. The `0.45`
   grid start is in MiniLM units and will select nothing, the same failure the
   margin grid hit.
2. Sweep candidate thresholds over that derived range, selecting on a stated
   module-level rule — macro-F1, or precision at a recall floor. **Register the
   rule before looking at what it selects.**
3. Report joint accuracy and macro-F1 on the test slice as the result.

## Acceptance

- The selected threshold is chosen on tuning-slice data only, and
  `evaluation_report.json` labels it `tuned_on: true` like the other two.
- Module joint accuracy on the test slice is greater than `0.0`.
- Module precision rises materially from `≈0.2` without recall collapsing.
- Test-slice **route** accuracy is unchanged at `0.7928` — this change must not
  be able to move the route, and a moved headline means `_emit_modules` leaked
  into the routing path.
- `docs/training-and-evaluation.md` drops "not yet re-derived" from the units
  trap and the known-limitations list.

## Out of scope

Acting on modules at serving time. They stay diagnostic-only; making them
route-affecting is a separate, larger decision.
