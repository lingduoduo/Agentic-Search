# Generated Context Pack

# Search Agent Deterministic Auto Search

## Sources

- [Specification: 2026-06-28-search-agent-deterministic-auto-search-design.md](../archive/specs/2026-06-28-search-agent-deterministic-auto-search-design.md)
- [Plan: 2026-06-28-search-agent-deterministic-auto-search.md](../archive/plans/2026-06-28-search-agent-deterministic-auto-search.md)

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

### Out of scope

- Changing `_force_final_answer` / the no-fabricate invariant.
- Wiring the M10 router into the auto-search retriever (PR B).

## Implementation Plan Context

### Overview

Spec: 2026-06-28-search-agent-deterministic-auto-search-design.md
Status: shipped (consolidated in PR #347).

**Goal:** guarantee retrieval fires at least once at the format-error dead-end,
gated behind `auto_search_on_deadend` (default `True`). Honest scope — retrieval
runs once; the answer still requires an `<answer>` tag (no-fabricate invariant).

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
