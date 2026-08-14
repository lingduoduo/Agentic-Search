# Plan: promotion readiness

Spec: `docs/superpowers/specs/2026-08-14-intent-promotion-readiness-design.md`

**This plan does not promote the router.** It produces the checklist, the
mechanism to answer the one criterion the eval sets cannot, the rollback
procedure, and a recorded decision.

## Steps

### 1. Re-measure on merged `main`

The checklist has to be written against what actually ships, not against
numbers from a branch.

→ verify: argmax `0.8159` CI `[0.757, 0.863]`, coverage `0.597`, served
accuracy `0.9667` CI `[0.917, 0.987]` on 120 answered with 4 wrong, hard_40
`0.7000` CI `[0.546, 0.819]`, out-of-scope AUC `0.8578` on 31 held-out probes.

### 2. Write the checklist, pre-registered

Five criteria, all of which must hold. The one that matters most is stated
against the **lower bound of the confidence interval**, not the point estimate —
which is the whole reason #526 had to come first.

→ verify: each criterion has a number, a current value, and a met/unmet verdict.

### 3. Shadow mode

`AGENTIC_SEARCH_INTENT_SHADOW_MODE`. Scores every auto-routed request, records
what the router *would* have decided, discards it, falls through to the
classifier exactly as if no index were configured.

→ verify: a vector that would be served confidently still reaches the LLM
classifier, *and* its prediction is recorded. Both halves asserted — recording
without the fall-through is a silent promotion; falling through without
recording gathers nothing.

→ verify: `route_shadow_*` fields are distinct from `route_predicted_intent`, so
a shadow run cannot be read back as a served one.

→ verify: off by default. Promotion-adjacent machinery must not arrive switched on.

### 4. Rollback, including the trap

→ verify: documented that the index cache is never invalidated **and a failed
load is cached too**, so neither enabling nor rolling back takes effect at
runtime. Any plan assuming a runtime toggle is wrong in both directions.

## The decision

**Not yet, on criterion 1.** The 95% interval on test-slice accuracy is
`[0.757, 0.863]` and the bar is `0.80` — the interval contains it, so the
evidence does not distinguish "clears the bar" from "does not". No amount of
tuning moves that; only more queries do.

Criterion 4 (production shadow data) is also unmet, and is the cheaper of the
two to close: shadow mode exists now and costs nothing to run, because the
router stays dark while it gathers.

Two of five criteria are met today — served accuracy and out-of-scope AUC, both
comfortably.

## Why this is a better "not yet" than the previous seven

Each earlier change deferred promotion without saying what would settle it. This
one names both blocking criteria, gives each a number, and ships the mechanism
that answers one of them without any request being routed by the model.
