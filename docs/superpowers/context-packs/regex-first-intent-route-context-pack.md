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

### Testing

- **`_regex_route` unit tests**, one behavior each: tool imperatives → TOOL; bare
  term + lookup imperative → SEARCH; question/explain/generative/trailing-`?` →
  CHAT; currency-conflict → None; genuinely ambiguous phrase → None.
- **`route_query` integration** (reuse `_FakeLLM` that records calls): a confident
  regex case returns the strategy and the **LLM is never consulted**
  (`llm.calls == []`); a non-confident case with an LLM **does** consult
  `classify_route`; the no-LLM path still resolves via `_rule_based_route`.
- **Regression:** update the `test_agent_router.py` cases that assumed the LLM
  classifier ran for now-deterministic inputs (e.g. `What is FAISS?` used to reach

…

## Implementation Plan Context

### Global Constraints

- `_regex_route` is high-precision: return a strategy ONLY on a confident match; return `None` on no-match or a currency/fact cross-cue conflict.
- Anchor tool/search/chat imperative cues to the START of the stripped query (`^`), so a command (`send an email`) differs from a description (`how to send an email`).
- Reuse the existing `_is_bare_lookup`; do NOT change `classify_route` or `_rule_based_route` behavior.
- No new dependencies. No change to `app.py` dispatch/degradation.
- Run `ruff check <files> --fix && ruff format <files>` before each commit (repo has a ruff pre-commit hook; if a commit aborts because the hook reformatted files, `git add -A` and re-run the same commit).

…

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
