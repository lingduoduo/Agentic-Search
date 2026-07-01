# SPEC — Move `src/agents/` framework modules into `core/`

## Objective

Group the four framework modules (`base.py`, `state.py`, `graph_base.py`,
`control_flow_trace.py`) under `src/agents/core/`, leaving the capability loop
packages (`generation/`, `search/`, `tool/`) and `components/` in place. **Zero
behavior change.** Stacked on PR #360 (capability grouping); rebase onto `main`
once #360 merges.

Name is `core/` — NOT `utils/`. These are the base class, the loop registry, the
shared state models, and the control-flow tracer: the framework spine, not a
grab-bag of helpers.

## Target structure

```
src/agents/
  __init__.py                    # public API preserved; base/state re-exports point at .core.*
  core/
    __init__.py                  # package marker (docstring only)
    base.py                      # AgentLoopBase, registry, config, RolloutStep, simple_timer
    state.py                     # AgentState, Plan, TaskNode, ToolCall, … (domain models)
    graph_base.py                # BaseAgent (pydantic + streaming)
    control_flow_trace.py        # ControlFlowRecorder, ControlFlowEvent, EventSink
  generation/  search/  tool/  components/   # unchanged
```

## Import strategy

Use **explicit module paths** — `from src.agents.core.base import X` — not a
`core/__init__.py` re-export list. `base`/`state` export dozens of symbols;
enumerating them in a re-export shim is fragile and error-prone. A pure
mechanical `agents.<mod>` → `agents.core.<mod>` rewrite is safer.

`core/__init__.py` is a docstring-only package marker.

## Migration mechanics

1. `git mv base.py state.py graph_base.py control_flow_trace.py core/`; add
   `core/__init__.py`.
2. The moved files need **no internal edits**: `base.py`'s
   `from .control_flow_trace import ControlFlowEvent` stays valid (both siblings
   in `core/`); `state.py` / `control_flow_trace.py` have no intra-package
   imports; `graph_base.py` imports only `src.internal.*` (absolute).
3. Mechanical rewrite across all files **except `src/agents/core/*`**:
   `agents.(base|state|graph_base|control_flow_trace)\b` → `agents.core.\1`.
   This catches `from src.agents.base import`, `from .agents.state import`
   (in `src/__init__.py`), the absolute `from src.agents.control_flow_trace
   import` in `search/search.py`, and the Sphinx `:class:` docstring refs in
   `src/training/…`.
4. Relative imports needing explicit handling (no `agents.` prefix):
   - `src/agents/__init__.py`: `from .base import` → `from .core.base import`;
     `from .state import` → `from .core.state import`.
   - `src/agents/components/*.py` (5 files): `from ..state import` →
     `from ..core.state import`.

## Scope

~90 sites (base 53, state 36, graph_base 3, control_flow_trace 5 deep imports;
6 docstring refs). No runtime `patch()` string targets reference these modules —
lower risk than the loops move despite the larger count.

## Registration & circular imports

No circular risk: `base → control_flow_trace` is the only intra-core edge and
`control_flow_trace` imports nothing from `agents`. Registration is unaffected
(the `@register` decorators live in the loop modules, not these four).

## Verification

- `python -c "import src.agents; from src.agents.core.base import list_registered_agent_loops as L; print(sorted(L()))"`
  → `['plain_generation', 'search_agent', 'single_turn_agent', 'tool_agent']`
- `from src.agents import AgentLoopBase, AgentState, SearchAgentLoop` (public API intact)
- `ruff check src tests examples`
- Full `pytest` (model-load env overrides) → 2260 passed, 2 skipped (baseline).

## Boundaries

- **Always**: preserve `from src.agents import X` public API and the registry;
  keep behavior identical; run full pytest.
- **Never**: name it `utils/`; add re-export shims at old paths; move
  `components/` or the loop packages; edit the moved files' logic.
