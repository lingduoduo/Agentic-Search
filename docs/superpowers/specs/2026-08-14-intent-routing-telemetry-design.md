# Make margin abstentions, modules, and the composite flag visible in production

## Status

Open since #511. Not started.

## Problem

Three routing signals are computed on every auto-routed request and then thrown
away outside the dev console.

**1. Modules and the composite flag never reach production telemetry.**
In `src/internal/servers/web/intent_routing.py`, `model_detail` is built with
both fields and handed to `_capture.record_stage(...)`, which only runs under
`AGENTIC_SEARCH_DEBUG_PANELS`. The adjacent `telemetry.update(...)` — the one
persisted with the session in production — receives five fields and neither of
these two:

```python
"modules": list(model_choice.modules),      # -> record_stage only
"composite": model_choice.composite,        # -> record_stage only

telemetry.update(
    route_predicted_intent=..., route_confidence=..., route_threshold=...,
    route_abstained=..., route_model_latency_ms=...,   # no modules, no composite
)
```

**2. Margin abstentions never reach `route_request` at all.**
`ml_intent.predict_route` returns `None` on `margin_below_threshold` after
recording its own capture stage. `route_request` cannot distinguish that from
"no index configured" — both are `model_choice is None` — so a margin abstention
is invisible even in the fields telemetry *does* carry.

## Why this matters more than it looks

**The composite flag exists solely to gather data.** Its docstring says so: it is
recorded so a future plan-aware router "can be designed against measured data
rather than guesses." A signal that is only observable under a dev-only debug
panel gathers nothing. The feature is, in its own stated purpose, inert.

The margin gate is also now the router's **only** working abstention — the
confidence floor at `0.30` cannot fire under e5 (measured in-scope `0.792`–`0.896`
against a `0.30` floor). Every deferral the router makes is a margin abstention,
and production cannot count them. "How often does the router defer, and on what?"
is unanswerable from production data today.

There is a measured reason to want this: on the test slice, abstention is *not*
uniform — `search` serves only 11 of 37 queries while `chat` serves 18 of 37 with
no errors. Whether that pattern holds on real traffic is exactly the question
this telemetry would answer, and it bears directly on the promotion decision.

## Approach

Both fixes need a signature change; that is the reason for the deferral and it is
a small change:

1. Extend the production telemetry payload with `route_modules` and
   `route_composite` alongside the existing five `route_*` fields.
2. Give `predict_route` a way to report margin abstention rather than collapsing
   it into `None` — return the decision with an abstention reason, or add an
   explicit out-parameter. `route_request` then records
   `route_fallback_reason="margin_below_threshold"` on the same path it already
   records `model_below_threshold`.

Keep the routing behavior identical. A margin abstention must still defer to the
LLM classifier; only its observability changes.

## Acceptance

- A margin-abstained request persists a distinguishable
  `route_fallback_reason` in session telemetry with the debug panels **off**.
- `route_modules` and `route_composite` are present on served model routes with
  the panels off.
- No change to which route any request receives — the existing routing tests pass
  unmodified, and a test asserts abstention still defers to the classifier.
- The known-limitation bullet in `docs/training-and-evaluation.md` is removed.

## Out of scope

Acting on the composite flag. This spec makes it observable; a plan-aware router
is a separate design that should be written against the data this produces.
