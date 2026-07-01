# Agents Capability Grouping — Implementation Plan

> **For agentic workers:** mechanical package refactor, zero behavior change. Execute in order; the existing test suite is the acceptance gate.

**Goal:** Move the 5 agent-loop modules into `generation/`, `search/`, `tool/` sub-packages; keep framework modules top-level; preserve public API and the loop registry.

**Global Constraints:** No behavior change. `from src.agents import X` keeps working. Registry resolves all 4 canonical loops. No shim modules at old paths. Framework modules (`base`, `state`, `graph_base`, `control_flow_trace`) do not move.

---

### Task 1: Create sub-packages and move files

- [ ] `git mv` `plain.py single_turn.py` → `generation/`; `search.py agentic_rag.py` → `search/`; `tool_calling.py` → `tool/`.
- [ ] Create `generation/__init__.py`, `search/__init__.py`, `tool/__init__.py` with the boundary re-exports from the spec.
- [ ] Verify: `git status` shows renames, not delete+add.

### Task 2: Fix moved files' internal imports

- [ ] In each of the 5 moved files, rewrite relative framework imports to absolute:
  - `from .base import ...` → `from src.agents.base import ...`
  - `from ..context... import ...` → `from src.context... import ...`
  - `from .state import ...` → `from src.agents.state import ...` (if present)
- [ ] Verify: `grep -n "from \.\." <moved files>` returns nothing; `python -c "import src.agents.generation, src.agents.search, src.agents.tool"` succeeds.

### Task 3: Update `src/agents/__init__.py`

- [ ] Point the loop imports at sub-packages: `from .generation import ...`, `from .search import ...`, `from .tool import ...`. Keep all `state`/`base` imports unchanged.
- [ ] Verify: `python -c "import src.agents; from src.agents.base import list_registered_agent_loops as L; print(sorted(L()))"` → `['plain_generation', 'search_agent', 'single_turn_agent', 'tool_agent']`.

### Task 4: Update external deep-import sites (~29)

- [ ] Rewrite each site (src/, tests/, examples/) to `from src.agents.<capability> import <symbol>` per the spec's list.
- [ ] **Also rewrite string-literal path references** — `patch("src.agents.<oldmodule>.X")` targets in tests are NOT import statements and are easy to miss. Grep `src\.agents\.(agentic_rag|tool_calling|plain|single_turn)\.` and rewrite `agentic_rag.` → `search.agentic_rag.`, `tool_calling.` → `tool.tool_calling.`, `plain.`/`single_turn.` → `generation.…`. Leave `src.agents.search.<Class>` strings — they still resolve via the package re-export.
- [ ] Verify: `grep -rn "from src.agents.\(plain\|single_turn\|search\|tool_calling\|agentic_rag\) import" src tests examples | grep -v "agents/\(generation\|search\|tool\)/"` returns nothing (old module-path imports gone), EXCEPT `from src.agents.search import` lines that now resolve to the package (acceptable — they hit `search/__init__`).

### Task 5: Verify + commit

- [ ] `ruff check src tests examples --fix && ruff format .`
- [ ] `pytest tests/unit/test_agent_loop.py tests/unit/test_agentic_rag.py tests/unit/test_run_agentic_search.py tests/unit/test_on_turn_callback.py tests/unit/test_loop_controller.py tests/unit/servers/web/test_loop_runners.py -q`
- [ ] Full `pytest` with model-load env overrides.
- [ ] Commit; push; open PR.
