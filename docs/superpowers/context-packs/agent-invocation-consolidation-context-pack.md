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
- **`resolve_agent_name` raises `KeyError` for non-registry names** (e.g. `chat_loop`, `search_tool`) so callers keep routing those on their existing paths — it never silently maps them.
- **No behavior change** to what any loop does, nor to web intent auto-detection (only the dispatch *target* for the two registry modes moves to the registry).

---

### Task 1: Canonical names + alias resolver

**Files:**
- Modify: `src/agents/base.py` (after `list_registered_agent_loops`, ~`:52`)
- Modify: `src/__init__.py` (alongside the other `from .agents.base import ...` exports, ~`:17`)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Produces: `CANONICAL_AGENT_NAMES: frozenset[str]`; `resolve_agent_name(name: str) -> str` (returns a canonical registry name; raises `KeyError` for non-registry names). Exported from `src`.

- [ ] **Step 1: Write the failing test**

```python

### tests/unit/test_agent_loop.py

from src.agents.base import resolve_agent_name, CANONICAL_AGENT_NAMES
import pytest

def test_resolve_cli_aliases():
    assert resolve_agent_name("single") == "single_turn_agent"
    assert resolve_agent_name("search") == "search_agent"
    assert resolve_agent_name("tool") == "tool_agent"

def test_resolve_canonical_names_passthrough():
    for name in ("plain_generation", "single_turn_agent", "search_agent", "tool_agent"):
        assert resolve_agent_name(name) == name

def test_resolve_rejects_non_registry_modes():
    for mode in ("chat_loop", "search_tool", "hybrid_search", "chat_once", "nope"):
        with pytest.raises(KeyError):
            resolve_agent_name(mode)

def test_canonical_names_are_registered():
    from src.agents.base import list_registered_agent_loops
    registered = set(list_registered_agent_loops())
    assert CANONICAL_AGENT_NAMES <= registered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_agent_loop.py -k "resolve or canonical_names_are_registered" -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_agent_name'`.

- [ ] **Step 3: Implement**

In `src/agents/base.py`, after `list_registered_agent_loops` (~`:52`):

```python

### Task 2: Route CLI dispatch through the registry

**Files:**
- Modify: `examples/run_agentic_search.py` (the `if args.mode ==` dispatch, ~`:1236`; and the loop-class references in `run_single_turn`/`run_search_agent`/`run_tool_agent`, ~`:768/852/944`)
- Test: `tests/unit/test_run_agentic_search.py`

**Interfaces:**
- Consumes: `resolve_agent_name`, `get_registered_agent_loop` (Task 1 + existing).
- Produces: CLI selects each loop class via `get_registered_agent_loop(resolve_agent_name(args.mode))`; the bespoke `run_*` helpers keep their per-loop construction.

- [ ] **Step 1: Write the failing test**

Read the existing `tests/unit/test_run_agentic_search.py` first to match its fixtures. Add a test that the CLI mode→class selection goes through the registry. If the file already drives `main()` with mocked model loading, mirror that; otherwise add a focused test of a small helper. Example shape (adapt to existing harness):

```python
def test_cli_mode_resolves_to_registry_class(monkeypatch):
    from src.agents.base import resolve_agent_name, get_registered_agent_loop
    from src.agents.search import SearchAgentLoop
    assert get_registered_agent_loop(resolve_agent_name("search")) is SearchAgentLoop
    from src.agents.tool_calling import ToolAgentLoop
    assert get_registered_agent_loop(resolve_agent_name("tool")) is ToolAgentLoop
```

- [ ] **Step 2: Run test to verify it fails (or is meaningful)**

Run: `pytest tests/unit/test_run_agentic_search.py -k registry -v`
Expected: FAIL if the helper/import is not yet present, or document that this asserts the resolver wiring the CLI will use.

- [ ] **Step 3: Implement**

_[Section compacted.]_

### Task 3: Route web `search_agent` + `tool_agent` dispatch through the registry

**Files:**
- Modify: `src/internal/servers/web/app.py` (`search_agent` block ~`:934`, `tool_agent` block ~`:996`)
- Test: the web app test path (`tests/unit/servers/web/` — find the existing dispatch/streaming test and mirror it)

**Interfaces:**
- Consumes: `resolve_agent_name`, `get_registered_agent_loop`.
- Produces: the two registry-mode blocks select their class via the registry; construction unchanged.

- [ ] **Step 1: Write the failing test**

Find the existing web dispatch test (e.g. `tests/unit/servers/web/test_sse_streaming.py` or an app test). Add a focused assertion that the `search_agent`/`tool_agent` modes resolve to the registry classes. If the web blocks are hard to unit-drive, assert the resolver mapping the app now uses:

```python
def test_web_registry_modes_map_to_classes():
    from src.agents.base import resolve_agent_name, get_registered_agent_loop
    from src.agents.search import SearchAgentLoop
    from src.agents.tool_calling import ToolAgentLoop
    assert get_registered_agent_loop(resolve_agent_name("search_agent")) is SearchAgentLoop
    assert get_registered_agent_loop(resolve_agent_name("tool_agent")) is ToolAgentLoop
```

- [ ] **Step 2: Run test to verify it fails/meaningful**

Run: `pytest tests/unit/servers/web/ -k registry -v`
Expected: FAIL until the resolver is wired (or document it pins the mapping the app uses).

- [ ] **Step 3: Implement**

_[Section compacted.]_

### Task 4: Documentation + fix stale CLAUDE.md reference

**Files:**
- Create: `docs/agent-invocation-surface.md`
- Modify: `.claude/CLAUDE.md` (the `custom.py` / `CustomAgent` reference)

**Interfaces:**
- Consumes: nothing (docs).

- [ ] **Step 1: Write the invocation-surface doc**

Create `docs/agent-invocation-surface.md` with one table covering every agent/mode. Columns: name · category (registry loop / non-registry loop / retrieval pipeline) · canonical name · CLI flag · web mode · scenario · construction notes. Include the scenario→agent mapping here (NOT as code — the web auto-detection is a non-goal to rework, so a consumed table would be dead data):

```markdown
| Mode | Category | Canonical | CLI | Web mode | Scenario |
|------|----------|-----------|-----|----------|----------|
| plain_generation | registry loop | plain_generation | (n/a) | — | smoke test |
| single_turn_agent | registry loop | single_turn_agent | --mode single | — | one-shot RAG |
| search_agent | registry loop | search_agent | --mode search | search_agent | multi-turn QA |
| tool_agent | registry loop | tool_agent | --mode tool | tool_agent | function calling |
| AgenticRAGLoop | non-registry loop | (none) | — | chat_loop | iterative RAG |
| search_tool | retrieval pipeline | (none) | — | search_tool | raw search |
| hybrid_search | retrieval pipeline | (none) | — | hybrid_search | hybrid retrieval |
| chat_once | retrieval pipeline | (none) | — | chat_once | single-shot RAG answer |
```

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
