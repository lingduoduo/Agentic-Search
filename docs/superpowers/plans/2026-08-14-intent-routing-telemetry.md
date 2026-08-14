# Plan: make routing signals visible in production

Spec: `docs/superpowers/specs/2026-08-14-intent-routing-telemetry-design.md`

## The signature question, decided

The spec says this "needs a `predict_route` or `route_request` signature
change." There are two ways, and the cheaper one is also the cleaner one.

**Rejected — add a `telemetry` parameter to `predict_route`.** `route_request`
would call `predict_route(query, settings=..., telemetry=...)`, and every test
double spelled `lambda q, settings=None: None` raises `TypeError` on the new
keyword. There are **eight** such doubles across `test_intent_routing.py`,
`test_agent_router.py`, and `test_stage_emits_intent.py`. That is a lot of churn
to buy a parameter, and this repo has been bitten before by an added optional
kwarg breaking strict test-double signatures.

**Chosen — carry the abstention in the return value.** `IntentModelDecision`
gains `margin` and `abstain_reason`. `predict_route` returns the decision on a
margin abstention instead of `None`, with `abstain_reason` set, and stops
recording its own capture stage. `route_request` then treats it exactly like the
confidence abstention it already handles.

No signature changes at all, so no test double moves. It also **unifies the two
abstention paths**: today confidence abstention returns a decision and margin
abstention returns `None`, which is the asymmetry that made margin abstention
invisible in the first place.

## Preserving the confidence path exactly

`abstain_reason` is populated **only** for `margin_below_threshold`. `decide()`
checks confidence first, so a confidence abstention already carries
`abstain_reason="confidence_below_threshold"` internally — but propagating that
would change `route_request`'s existing `fallback_reason` from
`model_below_threshold` to something new for a path this work is not about.
Leaving it `None` there keeps the confidence path byte-identical.

## Steps

### 1. `IntentModelDecision` gains `margin` and `abstain_reason`

Both default, so every existing construction in tests keeps working and means
"served, not abstained" — which is what those tests intend.

`margin` exists so the capture payload does not lose a field: the stage
`predict_route` records today includes `margin`, and that stage is going away.

→ verify: `test_stage_emits_intent.py` and `test_agent_router.py` construct
`IntentModelDecision` positionally/by keyword without the new fields and still pass.

### 2. `predict_route` returns the margin-abstained decision

Delete the early `record_stage(...)` + `return None` block. Set
`abstain_reason="margin_below_threshold"` on the returned decision instead.

→ verify: `predict_route` records **no** capture stage on any path — the
existing test that asserts `recorded == []` for the both-gates case now holds
for the margin-only case too, and `route_request` records the single stage.

### 3. `route_request` treats it as an abstention and fills telemetry

`abstained` becomes `abstain_reason is not None or confidence < threshold`;
`fallback_reason` prefers the decision's own reason. Add `route_modules` and
`route_composite` to the `telemetry.update(...)` call, and `margin` to the
capture detail.

→ verify: routing behavior is unchanged — a margin abstention still falls
through to the LLM classifier, which is the invariant that matters.

### 4. Tests

- a margin abstention persists `route_fallback_reason="margin_below_threshold"`
  in telemetry **with the debug panels off**
- `route_modules` / `route_composite` are present on a served model route
- a margin abstention still defers to the classifier (behavior unchanged)
- exactly one `intent_model` capture stage is recorded, never two

### 5. Docs

Remove the known-limitation bullet; record that abstention is now measurable.

## Risk

The one real risk is a double capture stage: `predict_route` recorded one on the
margin path precisely because it returned `None` and `route_request` never saw
it. Now that it returns a decision, `route_request` records one — so
`predict_route`'s must go, or every margin abstention emits two stages with
conflicting payloads. Step 4's last test exists for exactly that.
