# Generated Context Pack

# Regex First Intent Route

## Sources

- [Specification: 2026-07-05-regex-first-intent-route-design.md](../specs/2026-07-05-regex-first-intent-route-design.md)
- [Plan: 2026-07-05-regex-first-intent-route.md](../plans/2026-07-05-regex-first-intent-route.md)

## Specification Context

### Goal

Add a high-precision, anchored regex pass that runs **before** the LLM classifier.
Obvious cases are decided deterministically and for free; anything not confidently
matched falls through to the existing LLM classifier (or the existing lenient
rule-based fallback when no LLM is present). No new dependencies, no ML.

## Implementation Plan Context

### Task 1: `_regex_route` pure function + unit tests

**Files:**
- Modify: `src/internal/servers/web/intent_routing.py` (add regex constants after the existing `_TOOL_RE` block ~line 86, and add `_regex_route` after `_rule_based_route` ~line 129)
- Test: `tests/unit/servers/web/test_agent_router.py` (append)

**Interfaces:**
- Consumes: existing `RouteStrategy`, `_is_bare_lookup` (same module).
- Produces: `_regex_route(query: str) -> RouteStrategy | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/servers/web/test_agent_router.py`:

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/servers/web/test_agent_router.py -k regex_route -v`

…

### Task 2: wire `_regex_route` into `route_query`

**Files:**
- Modify: `src/internal/servers/web/intent_routing.py` (`route_query`, ~line 179)
- Test: `tests/unit/servers/web/test_agent_router.py` (append)

**Interfaces:**
- Consumes: `_regex_route` (Task 1), existing `classify_route`, `_rule_based_route`.
- Produces: no signature change to `route_query`; new behavior — confident regex short-circuits before the LLM.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/servers/web/test_agent_router.py`:

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/servers/web/test_agent_router.py -k "confident_regex or ambiguous_falls or currency_conflict" -v`

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
