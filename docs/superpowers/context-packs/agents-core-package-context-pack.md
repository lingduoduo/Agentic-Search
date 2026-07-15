# Generated Context Pack

# Agents Core Package

## Sources

- [Specification: 2026-07-01-agents-core-package-design.md](../specs/2026-07-01-agents-core-package-design.md)
- [Plan: 2026-07-01-agents-core-package.md](../plans/2026-07-01-agents-core-package.md)

## Specification Context

### Key decisions

- **`graph_base` stays a submodule, not re-exported by `core/__init__.py`.**
  Its `AgentState` TypedDict collides with `state.AgentState`. Re-exporting both
  would be ambiguous; `state.AgentState` is the one the facade already exposes.
  `graph_base` symbols are reached via `src.agents.core.graph_base.X`.
- **No shim modules at old paths.** All deep imports are rewritten to
  `src.agents.core.*` (same approach #360 used for the loops).
- **Zero behavior change.** `from src.agents import X` is unchanged; the facade
  simply points at `.core.*` internally. Existing test suite is the gate.

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
