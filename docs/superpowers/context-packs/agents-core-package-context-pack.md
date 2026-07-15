# Generated Context Pack

# Agents Core Package

## Sources

- [Specification: 2026-07-01-agents-core-package-design.md](../archive/specs/2026-07-01-agents-core-package-design.md)
- [Plan: 2026-07-01-agents-core-package.md](../archive/plans/2026-07-01-agents-core-package.md)

## Specification Context

### Problem

PR #360 grouped the 5 loop modules into `generation/`, `search/`, `tool/`
sub-packages but left the 4 framework modules loose at the top of
`src/agents/`. The result reads as lopsided: some children are packages, some
are bare files, with no rule distinguishing them at a glance.

#360 deferred moving `base.py` because of its ~18 importers ("triple the churn
for no grouping benefit"). But the churn is only mechanical import-path
rewrites, and the re-export facade neutralizes the public-API risk entirely.

## Implementation Plan Context

### Task 1: Move files into `core/`

- [x] `git mv base.py state.py graph_base.py control_flow_trace.py` → `core/`.
- [x] Verify: `git status` shows renames, not delete+add.

### Task 2: Write `core/__init__.py`

- [x] Re-export base + state + control_flow_trace symbols (per spec). Leave
  `graph_base` as a submodule — its `AgentState` collides with `state.AgentState`.

### Task 3: Rewrite deep imports (35 absolute sites)

- [x] `src.agents.{base,state,graph_base,control_flow_trace}` →
  `src.agents.core.\1` across `src/` and `tests/`. NB: BSD `sed` lacks `\b`;
  drop it (no module shares these prefixes).

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
