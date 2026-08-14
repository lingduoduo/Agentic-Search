# Plan: widen the evaluation instrument

Spec: `docs/superpowers/specs/2026-08-14-intent-wider-eval-instrument-design.md`

## Steps

### 1. Author the new queries and probes, score-free

90 clean eval queries (30 per route) and 36 out-of-scope probes (6 per
category), matching the existing informal business register. Authored before any
measurement, the same discipline as #523.

→ verify: schema valid, no id collisions, no duplicate text across *any* set, no
probe within `0.95` cosine of a canonical anchor.

→ result: two authored queries were rejected by the leakage guard and replaced.
`"who's on call this weekend"` scored `0.995` against the canonical `"who is on
call this weekend"`; `"send the offer letter to the new analyst"` scored `0.953`
against an anchor added in #524. Both would have measured the anchor rather than
the router.

### 2. Split the out-of-scope probes

`split_out_of_scope_probes` — stratified by the category in the probe id,
deterministic in the seed, halves disjoint and covering.

→ verify: 29 tuning / 31 reporting; the sweep tie-breaks on the tuning half and
`out_of_scope` is computed only against the reporting half, marked
`tuned_on: false`.

This closes the caveat carried since #512.

### 3. Publish confidence intervals

`wilson_interval`, reported beside every accuracy. Wilson rather than the normal
approximation because these slices are small and the proportions near `0.8`,
where the normal interval is too narrow and can exceed `1.0`.

→ verify: `accuracy_ci` and `served_accuracy_ci` present on every slice.

### 4. Re-run everything and re-derive the bars

→ verify: full suite green; bars re-pinned in the same commit.

## What re-running found, which the plan did not anticipate

**The sweep re-selected `top_k = 8`, not the shipped `15`.** Same pre-registered
rule, different tuning slice — seed 17 now samples 40 clean queries from 241
rather than 151, so it is a different 40.

This is the most useful thing the widening produced, and it is a result about
the *previous* work rather than about `k`: a value that moves from 15 to 8 when
the instrument grows was never determined at 15. The tuning curve shows `k=8`,
`15` and `25` within `0.014` of each other, with the tie-break toward lower `k`
deciding among them. On a 70-query tuning slice that gap is a couple of queries.

Shipped `k=8` per the rule, and documented it as **under-determined across
`8`–`25`** rather than as an optimum. That is a more defensible position than
the one it replaces.

## The AUC floor was lowered, deliberately

`0.85` → `0.83`. The convention says never lower a floor without recording why:
the number did not regress, the measurement changed. AUC is now computed against
probes no sweep has seen, where it was previously computed partly against probes
that had selected the thresholds it was measured at. `0.8578` on held-out probes
is the harder and more honest figure, and `0.83` restores the ~`0.02` headroom.

## What this does not fix

201 queries is better, not sufficient — the headline interval is still `±0.05`
and the `0.80` promotion bar sits inside it. `hard_40` is unchanged at 40 queries
with an interval around `[0.55, 0.82]`, which is why no bar is pinned to it.

Per the spec, **no tuning decision was re-opened on the strength of the new
instrument** beyond the hyperparameters the sweep re-selects by its existing
rule. Promotion in particular is untouched.
