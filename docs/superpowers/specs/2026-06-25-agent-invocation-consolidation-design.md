# Agent invocation consolidation — design

**Date:** 2026-06-25
**Status:** Approved scope (consolidate on registry + document); implementation
plan pending.
**Scope chosen:** route loop-backed entry points through the registry, add
canonical names + an alias map and a thin scenario→agent map, and document the
full invocation surface. **Not** a config-file rendering DSL (deferred — YAGNI for
the current agent count).

## Problem

The same agents are invoked through three surfaces with three divergent dispatch
mechanisms and naming vocabularies, and the one clean abstraction that should
unify them — the loop registry — is bypassed by both callers.

| Surface | Dispatch | Vocabulary | Uses registry? |
|---|---|---|---|
| Registry (`agents/base.py:30`) | `@register(name)` + `get_registered_agent_loop` | `plain_generation`, `single_turn_agent`, `search_agent`, `tool_agent` | — |
| CLI (`run_agentic_search.py:1236`) | hand-written `if args.mode ==` | `single`, `search`, `tool` | no |
| Web (`web/app.py:184`) | intent auto-detect + manual instantiation | `search_tool`, `hybrid_search`, `chat_once`, `chat_loop`, `search_agent`, `tool_agent` | no |

Consequences: three names for the same agent with nothing enforcing the mapping;
two hand-rolled dispatchers re-implementing what the registry already does;
`AgenticRAGLoop` is a loop that is **not** in the registry; CLAUDE.md references
`src/agents/custom.py` / `CustomAgent`, **which does not exist**.

## The honest boundary: loops vs. retrieval pipelines

The web "modes" are **not all agent loops** — half are retrieval pipelines and do
not belong under a loop registry. This boundary is load-bearing for the design:

| Web mode | Target | Category |
|---|---|---|
| `search_tool` | `src.tools.search_tool` (`app.py:813`) | retrieval pipeline |
| `hybrid_search` | `_run_hybrid_search` (`app.py:855`) | retrieval pipeline |
| `chat_once` | `answer_with_retrieval` (`app.py:481`) | retrieval pipeline |
| `chat_loop` | `AgenticRAGLoop` (`app.py:900`) | agent loop (**unregistered**) |
| `search_agent` | `SearchAgentLoop` (`app.py:934`) | agent loop (registered) |
| `tool_agent` | `ToolAgentLoop` (`app.py:996`) | agent loop (registered) |

**Design rule:** the registry consolidates **agent loops only**. Retrieval-pipeline
modes stay where they are and are *documented as a distinct category* — they are
not forced into the loop registry.

## Design

### 1. Registry is the single source of truth for agent loops

Both CLI and web resolve loop-backed entry points through
`get_registered_agent_loop(name)` (`agents/base.py:40`) instead of bespoke
`if/elif`. The hand-rolled dispatchers are deleted in favor of:

```python
loop_cls = get_registered_agent_loop(canonical_name)
loop = loop_cls(tokenizer, server_manager, config=...)
output = await loop.run(messages, sampling_params, on_turn=on_turn)
```

### 2. Register `AgenticRAGLoop` (close the gap)

`AgenticRAGLoop` (`agents/agentic_rag.py:101`) is loop-shaped but unregistered and
does not inherit `AgentLoopBase`. **Open item (verify in plan):** confirm it
conforms to the `run(messages, sampling_params, *, on_turn) -> AgentLoopOutput`
contract. If yes → add `@register("agentic_rag")`. If it diverges → either adapt
it to the contract or document it as a deliberate non-registry loop. Do **not**
register a class that violates the `dict[str, type[AgentLoopBase]]` contract.

### 3. Canonical names + alias map (don't break callers)

One canonical registry name per agent; existing CLI/web vocabularies become
documented aliases resolved to canonical names at the entry-point edge:

```
single        → single_turn_agent
search        → search_agent
tool          → tool_agent
chat_loop     → agentic_rag        (pending registration)
# unchanged canonical: plain_generation, search_agent, tool_agent
```

A tiny alias dict at each entry point (not a new framework). No public name is
removed; callers keep working.

### 4. Thin scenario→agent map

A dataclass/dict — **not** a YAML engine — mapping a scenario to a canonical agent
name plus config overrides, for the auto-detect/default paths:

```python
SCENARIO_AGENTS = {
    "qa_multi_turn":   ("search_agent", {...}),
    "function_call":   ("tool_agent",   {...}),
    "smoke_test":      ("plain_generation", {}),
}
```

This replaces the implicit scenario logic with one legible table. Retrieval-pipeline
modes are listed here too, marked as pipeline (not loop), so the table is the
*complete* scenario map even though pipelines dispatch differently.

### 5. Documentation

A single doc (`docs/` + a pointer from CLAUDE.md) with one table: each agent/mode →
category (loop vs pipeline) → canonical name → entry points (CLI flag, web mode) →
scenario → config. Fix the stale `custom.py` / `CustomAgent` reference in CLAUDE.md.

## Testing

- Registry resolution: every canonical name + alias resolves to the right class;
  unknown name raises a clear error (extends existing `get_registered_agent_loop`
  behavior).
- CLI dispatch parity: each `--mode` produces the same loop+config as today
  (golden-path test per mode).
- Web dispatch parity: `search_agent`/`tool_agent`/`chat_loop` route through the
  registry and behave identically; pipeline modes unchanged.
- `AgenticRAGLoop` registration (if it conforms): resolvable and runnable via the
  registry.

## Non-goals

- **Config-file (`agents.yaml`) rendering DSL.** Deferred — a registry + alias map +
  scenario table solves today's problem; a declarative agent-rendering system is
  speculative for ~5 loops and ~3 entry points. Revisit when the count grows.
- Consolidating retrieval-pipeline modes into the loop registry (different
  category by design).
- Changing any agent loop's behavior, or the `LoopController` control-flow work
  (fully orthogonal).
- Reworking web intent auto-detection logic (only its *dispatch target* moves to
  the registry; the detection stays).

## Relationship to other specs

- Orthogonal to `2026-06-25-agentic-search-loop-controller-design.md` (control
  flow) and `2026-06-25-tool-execution-mode-sketch.md` (action execution
  environment). This spec is about *which agent runs and how it's selected*, not
  how it loops or how its actions execute.
