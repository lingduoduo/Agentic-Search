# Agents `core/` Package — Implementation Plan

> **For agentic workers:** mechanical package refactor, zero behavior change.
> Execute in order; the existing test suite is the acceptance gate.

**Goal:** Move the 4 framework modules (`base`, `state`, `graph_base`,
`control_flow_trace`) into a `src/agents/core/` sub-package so the top level is
uniform; preserve public API and the loop registry.

**Global Constraints:** No behavior change. `from src.agents import X` keeps
working. No shim modules at old paths. Loop packages and `components/` untouched.

---

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
