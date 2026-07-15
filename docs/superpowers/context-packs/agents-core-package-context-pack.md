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

### Task 4: Rewrite relative imports (not caught by the absolute sweep)

- [x] `src/__init__.py`: `.agents.<mod>` → `.agents.core.<mod>`.
- [x] `src/training/{grpo,sft,reward}.py`: `..agents.<mod>` → `..agents.core.<mod>`.
- [x] `src/agents/components/*.py`: `..state` → `..core.state`.
- [x] Do NOT touch unrelated `base.py` modules under `src/tools/`,
  `src/internal/retrieval/backends/`, `src/internal/routing/construction/`.

### Task 5: Verify + commit

- [x] Import smoke test: `from src.agents import ...`, `from src.agents.core import ...`,
  `from src.agents.core.graph_base import BaseAgent`.
- [x] `ruff check src tests --fix && ruff format .` — all checks passed.
- [x] Full `pytest`: 2253 passed, 2 skipped. The 6 `test_execution_fallbacks.py`
  failures + 1 `test_graph_base` ordering flake are **pre-existing** — reproduced
  identically on `main` with `.env` present (real SEARCH_AGENT_MODEL load + no live
  retrieval server). 273/273 tests exercising the moved modules pass. See the
  "web test model-load gotcha": run these via `examples/run_web_integration_tests.sh`.
- [x] Commit; push; open PR.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
