# Generated Context Pack

# Intent Capture Parity

## Sources

- [Specification: 2026-07-05-intent-capture-parity-design.md](../specs/2026-07-05-intent-capture-parity-design.md)
- [Plan: 2026-07-05-intent-capture-parity.md](../plans/2026-07-05-intent-capture-parity.md)

## Specification Context

### Goal

Emit exactly **one** `intent` capture stage per request from a single point in
`route_query`, covering every decision path (explicit source, regex, LLM
classifier, no-LLM rule-based), so the Request Inspector always shows how a query
was routed and by which mechanism. Preserve the richer classifier detail
(`prompt`, `raw_label`) that exists today.

### Non-goals

- No change to routing *decisions* — this is observability only.
- `_regex_route` stays a pure `-> RouteStrategy | None` function (no side effects,
  its unit tests unchanged).
- No new capture infrastructure — reuse the existing `record_stage` /
  `request_capture` channel, which is a no-op when no capture is active.
- No frontend change — `RequestInspector` already renders any stage payload as
  JSON; the new `intent` payload surfaces automatically.

### Testing

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

## Implementation Plan Context

### Global Constraints

- One `intent` stage per request, emitted from `route_query`, labeled by mechanism (`explicit_source | regex | classifier | rule_based`); payload `{"mechanism", "strategy", **detail}`.
- The classifier path preserves `prompt` + `raw_label` in its detail; other paths use `{}`.
- `_regex_route` keeps its pure `-> RouteStrategy | None` signature (untouched).
- `record_stage` is a no-op when no capture is active — the hot path must stay unaffected when debug panels are off.
- No routing-decision changes; no frontend change (RequestInspector renders any payload).
- Run `ruff check <files> --fix && ruff format <files>` before committing (ruff pre-commit hook; if a commit aborts because the hook reformatted, `git add -A` and re-run the same commit).
- Branch: `feat/intent-capture-parity` (spec already committed there).

---

### Task 1: Centralize intent capture in `route_query`

**Files:**
- Modify: `src/internal/servers/web/intent_routing.py` (`classify_route`; add `_record_intent`; `route_query`)
- Test: `tests/unit/servers/web/test_stage_emits_intent.py` (rewrite), `tests/unit/servers/web/test_agent_router.py` (update `test_classify_route_*` to unpack the tuple)

**Interfaces:**
- Consumes: existing `_regex_route`, `_rule_based_route`, `_capture.record_stage`, `RouteStrategy`, `_FakeLLM` (in `test_agent_router.py`).
- Produces: `classify_route(query, llm) -> tuple[RouteStrategy, dict]`; `_record_intent(mechanism: str, strategy: RouteStrategy, detail: dict) -> None`.

- [ ] **Step 1: Rewrite `test_stage_emits_intent.py` to drive `route_query` (failing)**

Replace the entire contents of `tests/unit/servers/web/test_stage_emits_intent.py` with:

```python
from __future__ import annotations

from src.context.models import ChatMessage
from src.internal.servers.web import request_capture as rc
from src.internal.servers.web.intent_routing import route_query


class _FakeLLM:
    def complete(self, messages: list[ChatMessage], **_) -> str:
        return "search"


def _intent_stages():
    return [s for s in rc.active().stages if s.stage == "intent"]


def test_route_query_emits_regex_intent_stage():
    token = rc.start_capture("r", "What is FAISS?")
    try:
        strategy = route_query(
            "What is FAISS?", llm=_FakeLLM(), has_local_model=True, explicit_source=False
        )
        stages = _intent_stages()
        assert len(stages) == 1
        assert stages[0].label == "regex"
        assert stages[0].payload["mechanism"] == "regex"

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
