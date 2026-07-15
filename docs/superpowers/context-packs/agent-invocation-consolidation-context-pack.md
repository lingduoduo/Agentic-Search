# Generated Context Pack

# Agent Invocation Consolidation

## Sources

- [Specification: 2026-06-25-agent-invocation-consolidation-design.md](../specs/2026-06-25-agent-invocation-consolidation-design.md)
- [Plan: 2026-06-25-agent-invocation-consolidation.md](../plans/2026-06-25-agent-invocation-consolidation.md)

## Specification Context

### Testing

- Registry resolution: every canonical name + alias resolves to the right class;
  unknown name raises a clear error (extends existing `get_registered_agent_loop`
  behavior).
- CLI dispatch parity: each `--mode` produces the same loop+config as today
  (golden-path test per mode).
- Web dispatch parity: `search_agent`/`tool_agent`/`chat_loop` route through the
  registry and behave identically; pipeline modes unchanged.
- `AgenticRAGLoop` registration (if it conforms): resolvable and runnable via the
  registry.

### Non-goals

- **Config-file (`agents.yaml`) rendering DSL.** Deferred — a registry + alias map +
  scenario table solves today's problem; a declarative agent-rendering system is
  speculative for ~5 loops and ~3 entry points. Revisit when the count grows.
- Consolidating retrieval-pipeline modes into the loop registry (different
  category by design).
- Changing any agent loop's behavior, or the `LoopController` control-flow work
  (fully orthogonal).
- Reworking web intent auto-detection logic (only its *dispatch target* moves to
  the registry; the detection stays).

## Implementation Plan Context

### Global Constraints

- **Registry covers exactly four `AgentLoopBase` loops:** `plain_generation`, `single_turn_agent`, `search_agent`, `tool_agent`. `AgenticRAGLoop` (different constructor + `run` signature + return type) is NOT registered — it and the pipelines (`chat_loop`/`search_tool`/`hybrid_search`/`chat_once`) keep their existing dispatch untouched.
- **No public name removed.** Existing CLI flags (`single`/`search`/`tool`) and web modes keep working; they become documented aliases.
- **Construction stays per-loop.** The registry supplies the class; each call site constructs with its own config kwargs (`search_config=` / `config=` / `tools=,config=`). Do not attempt a uniform constructor.

…

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
