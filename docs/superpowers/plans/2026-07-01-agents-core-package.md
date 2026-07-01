# Agents Core Package — Implementation Plan

> Mechanical package refactor, zero behavior change. Stacked on PR #360. The test suite is the acceptance gate.

**Goal:** Move `base.py`, `state.py`, `graph_base.py`, `control_flow_trace.py` into `src/agents/core/`; update all imports; preserve public API + registry.

**Global Constraints:** No behavior change. `from src.agents import X` keeps working. Registry resolves all 4 loops. `core/` not `utils/`. Explicit `core.<mod>` module paths (no re-export shim). Do not edit moved files' logic or move `components/`/loop packages.

---

### Task 1: Move files + create package

- [ ] `git mv src/agents/{base,state,graph_base,control_flow_trace}.py src/agents/core/`.
- [ ] Create `src/agents/core/__init__.py` (docstring-only).
- [ ] Verify: `git status` shows 4 renames; `base.py`'s `from .control_flow_trace import ControlFlowEvent` is unchanged and still resolves (both in `core/`).

### Task 2: Mechanical rewrite (all files except `src/agents/core/*`)

- [ ] `grep -rl` the four module tokens and apply perl:
  `s/agents\.(base|state|graph_base|control_flow_trace)\b/agents.core.$1/g`
  over `src tests examples`, excluding `src/agents/core/`.
- [ ] Verify: `grep -rnE "agents\.(base|state|graph_base|control_flow_trace)\b" src tests examples | grep -v "agents.core." | grep -v "src/agents/core/"` → only relative-import lines from Task 3 remain (handled next).

### Task 3: Fix relative imports (no `agents.` prefix)

- [ ] `src/agents/__init__.py`: `from .base import` → `from .core.base import`; `from .state import` → `from .core.state import`.
- [ ] `src/agents/components/*.py` (5 files: search_tool, answer_generator, evidence_judge, planner, reranker_tool): `from ..state import` → `from ..core.state import`.
- [ ] Verify: `python -c "import src.agents"` succeeds.

### Task 4: Verify registry + public API

- [ ] `python -c "import src.agents; from src.agents.core.base import list_registered_agent_loops as L; print(sorted(L())); from src.agents import AgentLoopBase, AgentState, SearchAgentLoop; print('API OK')"` → all 4 loops + `API OK`.

### Task 5: Lint, full-suite gate, commit, PR

- [ ] `ruff check src tests examples` (+ `ruff format` any deltas).
- [ ] Full `pytest` with model-load env overrides → **2260 passed, 2 skipped**.
- [ ] Commit; push; open PR with base = `refactor/agents-capability-grouping` (stacked; retarget to `main` after #360 merges).
