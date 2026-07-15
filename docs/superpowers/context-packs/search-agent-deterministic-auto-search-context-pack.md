# Generated Context Pack

# Search Agent Deterministic Auto Search

## Sources

- [Specification: 2026-06-28-search-agent-deterministic-auto-search-design.md](../specs/2026-06-28-search-agent-deterministic-auto-search-design.md)
- [Plan: 2026-06-28-search-agent-deterministic-auto-search.md](../plans/2026-06-28-search-agent-deterministic-auto-search.md)

## Specification Context

### Goal

Guarantee `SearchAgentLoop` runs **at least one retrieval round** even when the
policy model never emits a recognized `<search>` tag — so retrieval is not
entirely LLM-gated.

### Scope: what this guarantees (and what it does not)

It guarantees only that **retrieval fires at least once** — *not* a non-empty
answer. The final answer still flows through `_force_final_answer`, which requires
an `<answer>` tag and otherwise returns `None`: the **no-fabricate invariant** (a
model refusal is never passed off as an answer; see
`test_forced_turn_emitting_no_answer_returns_none`). Therefore:

- A model that emits `<answer>` when prompted with the retrieved evidence now
  produces a *grounded* answer instead of dead-ending empty (realistic weak model;
  covered by `test_deadend_forces_answer_from_evidence`).
- A genuinely tag-less model (no tag on any turn, including the forced turn) still

…

### Tests

- `test_search_agent_loop_auto_searches_when_model_emits_no_action` — genuinely
  tag-less model: asserts `search_rounds == 1` + an `auto_search` event +
  `final_answer is None` (honest guarantee, no fabrication).
- `test_search_agent_loop_auto_search_disabled_preserves_format_recovery` — flag
  off ⇒ retrieval never runs.
- Legacy `…stops_after_repeated_no_action_turns` and
  `deadend_with_no_evidence_does_not_fabricate` pin the flag off to keep
  validating the format-error-stop / no-fabricate machinery.

### Out of scope

- Changing `_force_final_answer` / the no-fabricate invariant.
- Wiring the M10 router into the auto-search retriever (PR B).

## Implementation Plan Context

### Tasks

1. **Config flag.** `SearchAgentLoopConfig.auto_search_on_deadend: bool = True`.
2. **Trigger (in `SearchAgentLoop.run`, the `if not actions:` branch).** Before
   the format-recovery path, when `auto_search_on_deadend and state.search_rounds
   == 0 and question and consecutive_format_errors + 1 >=
   max_consecutive_format_errors`: set `actions = [(cfg.search_tag, question)]`,
   reset the format-error counter, and record an `auto_search` event. Otherwise
   fall through to the existing `_handle_no_action` path.
3. **Tests (honest).**
   - `…auto_searches_when_model_emits_no_action`: genuinely tag-less model ⇒
     `search_rounds == 1`, `auto_search` event, `final_answer is None`.

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
