# Intent-capture parity for regex-first routing — design

## Problem

The Dev Console Request Inspector records an `intent` capture stage only inside
`classify_route` (the LLM classifier). After regex-first routing landed
(`_regex_route` runs before the classifier), confident regex-decided queries — a
large fraction of real traffic (`What is FAISS?`, tool imperatives, bare lookups,
`?`-terminated questions) — never reach `classify_route`, so **no `intent` stage
is captured for them**. The Inspector shows the search/llm/final stages but a
blank where the routing decision should be, for exactly the queries a developer
most wants to understand.

## Goal

Emit exactly **one** `intent` capture stage per request from a single point in
`route_query`, covering every decision path (explicit source, regex, LLM
classifier, no-LLM rule-based), so the Request Inspector always shows how a query
was routed and by which mechanism. Preserve the richer classifier detail
(`prompt`, `raw_label`) that exists today.

## Non-goals

- No change to routing *decisions* — this is observability only.
- `_regex_route` stays a pure `-> RouteStrategy | None` function (no side effects,
  its unit tests unchanged).
- No new capture infrastructure — reuse the existing `record_stage` /
  `request_capture` channel, which is a no-op when no capture is active.
- No frontend change — `RequestInspector` already renders any stage payload as
  JSON; the new `intent` payload surfaces automatically.

## Approach — centralize the emit in `route_query`

`route_query` becomes the single intent-emit point. `classify_route` stops
emitting its own stage and instead **returns its detail** so `route_query` can
record it uniformly.

### `classify_route` signature change

Today: `classify_route(query, llm) -> RouteStrategy` and it calls
`record_stage("intent", "classify_route", {prompt, raw_label, strategy})`
internally.

New: `classify_route(query, llm) -> tuple[RouteStrategy, dict]`, returning
`(strategy, {"prompt": prompt, "raw_label": content})`. It no longer calls
`record_stage`.

### `route_query` emit

After the strategy is decided by any path, record one stage:

```python
def _record_intent(mechanism: str, strategy: RouteStrategy, detail: dict) -> None:
    _capture.record_stage(
        "intent", mechanism, {"mechanism": mechanism, "strategy": strategy.value, **detail}
    )
```

Wired per path:

- `explicit_source` → `_record_intent("explicit_source", SEARCH, {})`
- regex match → `_record_intent("regex", regex_choice, {})`
- LLM classifier → `strategy, detail = classify_route(...)`; `_record_intent("classifier", strategy, detail)`
- no-LLM rule-based → `_record_intent("rule_based", strategy, {})`
- classifier error → falls to `_rule_based_route`, recorded as `rule_based`

`_record_intent` is a no-op when no capture is active (via `record_stage`), so the
hot path is unaffected when debug panels are off.

### Stage label

The `record_stage` `label` argument becomes the `mechanism`, so the Inspector
renders `intent · regex`, `intent · classifier`, etc. — immediately showing which
mechanism decided.

## Data flow / example

`What is FAISS?` → `_regex_route` returns CHAT → `route_query` records
`intent · regex {mechanism: "regex", strategy: "chat"}` and returns CHAT, no LLM
call. Previously: no intent stage at all.

`the procurement approval flow` → `_regex_route` returns None →
`classify_route` returns `(CHAT, {prompt, raw_label})` → `route_query` records
`intent · classifier {mechanism: "classifier", strategy: "chat", prompt, raw_label}`.

## Testing

- **`classify_route` unit tests** (`test_agent_router.py::test_classify_route_*`):
  update to unpack the new tuple return (`strategy, _ = classify_route(q, llm)`).
- **`test_stage_emits_intent.py`**: currently asserts `classify_route` emits the
  intent stage; repoint it to drive `route_query` and assert the stage is emitted
  there (mechanism `classifier`, with `prompt`/`raw_label`).
- **New `route_query` capture tests** (under an active capture, one per mechanism):
  - explicit source → intent stage `mechanism == "explicit_source"`
  - regex-decided query (`What is FAISS?`) → `mechanism == "regex"`, `strategy == "chat"`
  - ambiguous query → `mechanism == "classifier"`, payload has `prompt` and `raw_label`
  - no-LLM path → `mechanism == "rule_based"`
  - no active capture → no error, nothing recorded
- **e2e capture test** (`test_request_capture_e2e.py`): stays green; since every
  query now yields an intent stage, it no longer depends on choosing a
  classifier-bound query (may keep its current query — still valid).
- Full suite green; ruff clean.

## Success criteria

- Every `/api/agent` request captures exactly one `intent` stage, labeled by
  mechanism, visible in the Request Inspector — including regex-decided queries.
- The classifier path still captures `prompt` + `raw_label`.
- Routing decisions and the `_regex_route` signature are unchanged.
