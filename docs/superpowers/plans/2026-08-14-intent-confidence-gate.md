# Plan: re-derive or retire the confidence gate

Spec: `docs/superpowers/specs/2026-08-14-intent-dead-confidence-gate-design.md`

## Pre-registered rule

**Committed before the sweep runs.**

> Extend `_SWEEP_MIN_CONFIDENCES` to span the score range actually observed on
> the tuning slice. Keep the existing selection rule **unchanged**: highest
> tuning served accuracy at coverage ≥ `0.60`, ties broken toward higher
> out-of-scope deferral, then toward lower `top_k`.
>
> **If the selected `min_confidence` is below the lowest in-scope score on the
> tuning slice — i.e. inert — the parameter is retired.** The sweep will then
> have said that no value which can actually fire beats one that cannot.

This deliberately does not invent a new objective. The existing rule already
prefers higher out-of-scope deferral when served accuracy ties, which is exactly
the property a working confidence gate would provide. If a live floor is worth
having, that rule will pick it without being told to.

Letting the incumbent rule decide also means the outcome cannot be steered by
choosing a favourable objective after seeing the curve.

## What the data already says about the odds

Measured on the tuning slice at the shipped `top_k=8`:

```
in-scope  n=70   0.7905–0.8682   p5 0.8075
probes    n=29   0.7610–0.8409   p95 0.8318
21 of 29 probes score above the lowest in-scope query
```

The ranges overlap heavily, so no floor cleanly separates them. But **8 of 29
probes fall below the in-scope minimum**, so a floor near `0.79` could reject
roughly a quarter of out-of-scope traffic at zero measured in-scope cost on this
slice. Whether that survives the selection rule — and whether it holds on the
test slice — is what the sweep decides.

## Steps

### 1. Extend the grid

The current grid is `(0.30 … 0.55)`, entirely below the observed range, so every
value is a no-op and the sweep cannot distinguish them. Extend it to span the
tuning range, keeping the existing low values so the sweep still records the
inert baseline it is being compared against.

→ verify: the sweep's row count matches the new grid size, and at least one row
carries a `min_confidence` that fires.

### 2. Run and apply the rule

→ verify: the selected value, and whether it is live or inert.

### 3a. If live — ship it

Update the default in all four places, re-derive the bars, and report the
held-out cost: how much test-slice coverage the floor removes, and how many
reporting-half probes it rejects.

### 3b. If inert — retire the parameter

Remove `intent_model_min_confidence` from `AppSettings`, `DEFAULT_CONFIG`, the
`decide()` confidence branch, `IntentModelDecision.threshold`, and both docs.

→ verify: **test-slice accuracy unchanged at `0.8159`**. Removing an unreachable
branch must be behaviour-preserving to the digit; a moved headline means it was
reachable after all and the removal is wrong.

## Risk

Low either way, but asymmetric. Shipping a live floor changes what the router
answers, so it needs the held-out cost measured. Retiring an inert one changes
nothing at runtime — the invariant in 3b is what proves that rather than
assuming it.
