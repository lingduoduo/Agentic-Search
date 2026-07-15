# Agents `core/` Package — Design

## Problem

PR #360 grouped the 5 loop modules into `generation/`, `search/`, `tool/`
sub-packages but left the 4 framework modules loose at the top of
`src/agents/`. The result reads as lopsided: some children are packages, some
are bare files, with no rule distinguishing them at a glance.

```
src/agents/
  base.py  state.py  graph_base.py  control_flow_trace.py   # loose framework files
  components/  generation/  search/  tool/                   # packages
```

#360 deferred moving `base.py` because of its ~18 importers ("triple the churn
for no grouping benefit"). But the churn is only mechanical import-path
rewrites, and the re-export facade neutralizes the public-API risk entirely.

## Target structure

```
src/agents/
  __init__.py          # public API facade unchanged; internal imports point at .core.*
  core/                # framework primitives — NOT loops
    __init__.py        # re-exports base + state + control_flow_trace symbols
    base.py
    state.py
    graph_base.py
    control_flow_trace.py
  components/           # reusable building blocks (unchanged)
  generation/  search/  tool/   # loops (unchanged)
```

Top level is now uniform: one facade + five packages, each with one role —
`core/` (built *from*), `components/` (built *with*), and the three loop
packages.

## Key decisions

- **`graph_base` stays a submodule, not re-exported by `core/__init__.py`.**
  Its `AgentState` TypedDict collides with `state.AgentState`. Re-exporting both
  would be ambiguous; `state.AgentState` is the one the facade already exposes.
  `graph_base` symbols are reached via `src.agents.core.graph_base.X`.
- **No shim modules at old paths.** All deep imports are rewritten to
  `src.agents.core.*` (same approach #360 used for the loops).
- **Zero behavior change.** `from src.agents import X` is unchanged; the facade
  simply points at `.core.*` internally. Existing test suite is the gate.

## Boundary symbols (`core/__init__.py` re-exports)

- from `base`: `AgentLoopBase`, `AgentLoopConfig`, `AgentLoopOutput`,
  `OnTurnCallback`, `RolloutStep`, `register`, `simple_timer`
- from `control_flow_trace`: `ControlFlowEvent`, `ControlFlowRecorder`,
  `EventSink`
- from `state`: `AgentState`, `PerformanceMetrics`, `Plan`, `PlanStep`,
  `RetrievedDocument`, `RouteDecision`, `TaskNode`, `TaskStatus`, `TaskType`,
  `ToolCall`, `ToolExecutionResult`, `ToolResult`, `ToolType`, `UserRequest`
