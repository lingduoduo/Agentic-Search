# Generated Context Pack

# Agent Invocation Consolidation

## Sources

- [Specification: 2026-06-25-agent-invocation-consolidation-design.md](../specs/2026-06-25-agent-invocation-consolidation-design.md)
- [Plan: 2026-06-25-agent-invocation-consolidation.md](../plans/2026-06-25-agent-invocation-consolidation.md)

## Specification Context

### Overview

**Date:** 2026-06-25
**Status:** Approved scope (consolidate on registry + document); implementation
plan pending.
**Scope chosen:** route loop-backed entry points through the registry, add
canonical names + an alias map and a thin scenario→agent map, and document the
full invocation surface. **Not** a config-file rendering DSL (deferred — YAGNI for
the current agent count).

## Implementation Plan Context

### Task 1: Canonical names + alias resolver

**Files:**
- Modify: `src/agents/base.py` (after `list_registered_agent_loops`, ~`:52`)
- Modify: `src/__init__.py` (alongside the other `from .agents.base import ...` exports, ~`:17`)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Produces: `CANONICAL_AGENT_NAMES: frozenset[str]`; `resolve_agent_name(name: str) -> str` (returns a canonical registry name; raises `KeyError` for non-registry names). Exported from `src`.

- [ ] **Step 1: Write the failing test**

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_agent_loop.py -k "resolve or canonical_names_are_registered" -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_agent_name'`.

…

### Task 2: Route CLI dispatch through the registry

**Files:**
- Modify: `examples/run_agentic_search.py` (the `if args.mode ==` dispatch, ~`:1236`; and the loop-class references in `run_single_turn`/`run_search_agent`/`run_tool_agent`, ~`:768/852/944`)
- Test: `tests/unit/test_run_agentic_search.py`

**Interfaces:**
- Consumes: `resolve_agent_name`, `get_registered_agent_loop` (Task 1 + existing).
- Produces: CLI selects each loop class via `get_registered_agent_loop(resolve_agent_name(args.mode))`; the bespoke `run_*` helpers keep their per-loop construction.

- [ ] **Step 1: Write the failing test**

…

### Task 3: Route web `search_agent` + `tool_agent` dispatch through the registry

**Files:**
- Modify: `src/internal/servers/web/app.py` (`search_agent` block ~`:934`, `tool_agent` block ~`:996`)
- Test: the web app test path (`tests/unit/servers/web/` — find the existing dispatch/streaming test and mirror it)

**Interfaces:**
- Consumes: `resolve_agent_name`, `get_registered_agent_loop`.
- Produces: the two registry-mode blocks select their class via the registry; construction unchanged.

- [ ] **Step 1: Write the failing test**

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
