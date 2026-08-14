# Retire or re-derive the confidence gate that cannot fire

## Status

Not started. Small, self-contained, and the cheapest way to remove a piece of
config that reads as a safety control but is not one.

## Problem

`AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE` defaults to `0.30`. It is a cosine
similarity, inherited from the MiniLM era, and it was never re-derived when the
encoder changed or when `top_k` moved to 15.

Measured at the current shipped settings over all 135 scored queries (111
test-slice + 24 out-of-scope probes):

```
floor 0.30    in-scope 0.7756–0.8567    out-of-scope 0.7674–0.8289
queries scoring below the floor: 0 / 135
```

**It cannot fire for any input, in scope or out.** `decide()` still checks it
first and can still return `abstain_reason="confidence_below_threshold"`, so the
code path exists, is tested, and is documented — it is simply unreachable at the
shipped value.

## Why this is worth a change rather than a comment

The training-and-evaluation guide already documents that the gate never fires,
so nobody is being actively misled by the doc. The risk is in the *config
surface*: an operator reading `configuration.md` sees a threshold whose stated
purpose is "reject requests nothing resembles", and reasonably concludes that
out-of-scope protection is configurable there. It is not. The only working
abstention is `min_route_margin`.

A knob that looks like a safety control and is inert is worse than no knob,
because it invites tuning that will do nothing and mask the absence of the
control it appears to provide.

## The two options, and why the choice needs data

**Re-derive it** on the tuning slice in the current encoder's units, the way
`min_margin` and `min_module_score` were. This keeps a second, absolute-score
abstention alongside the relative-margin one.

The catch is that the in-scope and out-of-scope ranges **overlap**
(`0.7756`–`0.8567` against `0.7674`–`0.8289`), so no single value separates them.
Any derived floor trades in-scope coverage for out-of-scope rejection along a
curve that must be measured before a value is picked. It may well turn out that
no value is worth shipping — which is itself a result.

**Retire it** — remove the parameter, the config key, and the confidence branch
in `decide()`, leaving margin as the sole abstention. Smaller surface, one fewer
inert knob, and one fewer thing to re-derive on every encoder change. The cost
is that a future encoder with well-separated absolute scores would have to
re-introduce it.

**This spec does not pre-judge which.** It requires the sweep to be run and the
choice recorded with its evidence, the same way `top_k` was settled.

## Approach

1. Sweep candidate floors on the **tuning slice** over the observed score range,
   recording in-scope coverage lost against out-of-scope probes rejected.
2. Register the decision rule before looking at the result.
3. If a value clears the rule, ship it and re-measure the held-out numbers. If
   none does, retire the parameter.

## Acceptance

- Either a derived, non-inert default with the sweep in `evaluation_report.json`,
  or the parameter removed from `AppSettings`, `DEFAULT_CONFIG`, `decide()`, and
  both docs.
- **Test-slice route accuracy unchanged at `0.8108`** if retiring — removing an
  unreachable branch must be behaviour-preserving, and a moved headline means it
  was reachable after all.
- `configuration.md` no longer documents a threshold that cannot fire.

## Out of scope

`min_route_margin` and `min_module_score`, both already derived. This is only
about the third gate.
