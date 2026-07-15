# Generated Context Pack

# Agents Capability Grouping

## Sources

- [Specification: 2026-07-01-agents-capability-grouping-design.md](../specs/2026-07-01-agents-capability-grouping-design.md)
- [Plan: 2026-07-01-agents-capability-grouping.md](../plans/2026-07-01-agents-capability-grouping.md)

## Specification Context

### Verification

- `python -c "import src.agents; from src.agents.base import list_registered_agent_loops as L; print(sorted(L()))"`
  → `['plain_generation', 'search_agent', 'single_turn_agent', 'tool_agent']`
- `ruff check src tests examples`
- Targeted: `pytest tests/unit/test_agent_loop.py tests/unit/test_agentic_rag.py tests/unit/test_run_agentic_search.py tests/unit/test_on_turn_callback.py tests/unit/test_loop_controller.py`
- Web loop runners: `pytest tests/unit/servers/web/test_loop_runners.py`
- Full: `pytest` (with the documented model-load env overrides).

## Implementation Plan Context

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

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
