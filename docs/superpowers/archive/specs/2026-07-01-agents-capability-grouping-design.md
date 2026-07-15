# SPEC — Group `src/agents/` loops into capability sub-packages

## Objective

Reorganize the five agent-loop modules in `src/agents/` into capability-named
sub-packages, with **zero behavior change**. Framework modules stay at the top
level. The public API (`from src.agents import X`) is preserved; deep import
sites are updated to the new paths.

Users: developers navigating the agent codebase. Success = same runtime
behavior, all tests green, cleaner package layout, and the loop registry still
resolves all four canonical loops.

## Target structure

```
src/agents/
  __init__.py                    # public API unchanged; internal imports point at sub-packages
  base.py  state.py  graph_base.py  control_flow_trace.py   # framework — stay top-level
  components/                    # unchanged
  generation/
    __init__.py                  # re-exports Plain*, SingleTurn*
    plain.py
    single_turn.py
  search/
    __init__.py                  # re-exports SearchAgentLoop(+Config), TurnControl,
    search.py                    #   build_search_agent_instruction, AgenticRAG{Loop,Config,Result}
    agentic_rag.py
  tool/
    __init__.py                  # re-exports ToolAgentLoop(+Config)
    tool_calling.py
```

Framework modules (`base`, `state`, `graph_base`, `control_flow_trace`) are NOT
agents and are NOT moved — moving `base.py` (18 importers) would triple the
churn for no grouping benefit.

## Boundary symbols (drive the sub-package re-exports)

- `generation/__init__.py`: `PlainGenerationLoop`, `PlainGenerationLoopConfig`,
  `SingleTurnAgentLoop`, `SingleTurnAgentLoopConfig`
- `search/__init__.py`: `SearchAgentLoop`, `SearchAgentLoopConfig`, `TurnControl`,
  `build_search_agent_instruction`, `AgenticRAGLoop`, `AgenticRAGConfig`,
  `AgenticRAGResult`
- `tool/__init__.py`: `ToolAgentLoop`, `ToolAgentLoopConfig`

## Migration mechanics

1. `git mv` the 5 loop files into the 3 new packages; add 3 `__init__.py`
   re-export files.
2. In the 5 moved files, rewrite intra-package **relative** imports
   (`from .base import`, `from ..context...`) to **absolute** (`from
   src.agents.base import`, `from src.context... import`) — dot-depth-proof.
3. Update `src/agents/__init__.py` to import from the sub-packages (keeps
   registration + public API intact).
4. Update every external deep-import site to `from src.agents.<capability>
   import <symbol>` (uniform rule; sub-package `__init__` re-exports make every
   currently-imported symbol reachable).

## Deep-import sites to update (~29)

- `src/training/data.py`, `src/training/ppo/search_agent_grpo_trainer.py`
- `src/internal/servers/web/app.py` (agentic_rag, search, tool_calling)
- tests: `test_run_agentic_search`, `test_on_turn_callback`, `test_agent_loop`,
  `test_execution_fallbacks`, `test_loop_controller`, `test_agentic_rag`,
  `servers/web/test_sse_streaming`, `servers/web/test_web_experience_app`,
  `servers/web/test_loop_runners`
- examples: `run_sft_grpo`, `run_bamboogle_eval`, `run_feedback_grpo`,
  `evaluate_bamboogle`, `run_retriever_aware_grpo`

## Registration

`@register("...")` fires on module import. `src/agents/__init__.py` imports the
sub-packages (which import their modules), so `plain_generation`,
`single_turn_agent`, `search_agent`, `tool_agent` all still register.

## Verification

- `python -c "import src.agents; from src.agents.base import list_registered_agent_loops as L; print(sorted(L()))"`
  → `['plain_generation', 'search_agent', 'single_turn_agent', 'tool_agent']`
- `ruff check src tests examples`
- Targeted: `pytest tests/unit/test_agent_loop.py tests/unit/test_agentic_rag.py tests/unit/test_run_agentic_search.py tests/unit/test_on_turn_callback.py tests/unit/test_loop_controller.py`
- Web loop runners: `pytest tests/unit/servers/web/test_loop_runners.py`
- Full: `pytest` (with the documented model-load env overrides).

## Boundaries

- **Always**: preserve `from src.agents import X` public API; keep the registry
  resolving all 4 loops; run ruff + targeted + full pytest; feature branch;
  spec + plan committed on branch.
- **Never**: change any loop behavior; move framework modules; add back-compat
  shim modules at old paths (the chosen strategy is to update import sites, not
  leave forwarding cruft); rename symbols.

## Known follow-on

PR #359 (agentic-rag loop optimization) edits `agentic_rag.py` in place on a
separate branch. Whichever merges second will need a rebase — a rename+edit
reconciliation on that one file. Flagged, not blocking.
