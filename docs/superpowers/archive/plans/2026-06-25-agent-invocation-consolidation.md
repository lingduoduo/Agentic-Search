# Agent Invocation Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the loop registry the single source of truth for which agent loop runs, reconcile the CLI/web naming vocabularies via a canonical-name + alias resolver, route CLI + web class selection through the registry, and document the full invocation surface.

**Architecture:** A small `resolve_agent_name()` + alias map in `src/agents/base.py` (alongside the existing `register`/`get_registered_agent_loop`) maps CLI (`single`/`search`/`tool`) and web (`search_agent`/`tool_agent`) names to the four canonical registry loops. The CLI dispatch and the web `search_agent`/`tool_agent` blocks select their loop **class** via `get_registered_agent_loop(resolve_agent_name(mode))` instead of hard-coded imports; per-loop construction (different config kwargs) stays at the call site. `AgenticRAGLoop` and the retrieval-pipeline modes are documented as a distinct non-registry category and left untouched.

**Tech Stack:** Python 3, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-25-agent-invocation-consolidation-design.md`

## Global Constraints

- **Registry covers exactly four `AgentLoopBase` loops:** `plain_generation`, `single_turn_agent`, `search_agent`, `tool_agent`. `AgenticRAGLoop` (different constructor + `run` signature + return type) is NOT registered — it and the pipelines (`chat_loop`/`search_tool`/`hybrid_search`/`chat_once`) keep their existing dispatch untouched.
- **No public name removed.** Existing CLI flags (`single`/`search`/`tool`) and web modes keep working; they become documented aliases.
- **Construction stays per-loop.** The registry supplies the class; each call site constructs with its own config kwargs (`search_config=` / `config=` / `tools=,config=`). Do not attempt a uniform constructor.
- **`resolve_agent_name` raises `KeyError` for non-registry names** (e.g. `chat_loop`, `search_tool`) so callers keep routing those on their existing paths — it never silently maps them.
- **No behavior change** to what any loop does, nor to web intent auto-detection (only the dispatch *target* for the two registry modes moves to the registry).

---

## File Structure

- **Modify** `src/agents/base.py` — add `CANONICAL_AGENT_NAMES`, `_AGENT_ALIASES`, `resolve_agent_name()`.
- **Modify** `src/__init__.py` — export `resolve_agent_name`.
- **Modify** `examples/run_agentic_search.py` — CLI dispatch resolves mode and selects the class via the registry.
- **Modify** `src/internal/servers/web/app.py` — `search_agent` + `tool_agent` blocks select the class via the registry.
- **Create** `docs/agent-invocation-surface.md` — the full invocation-surface table (loops vs pipelines, canonical names, aliases, entry points, scenario mapping).
- **Modify** `.claude/CLAUDE.md` — fix the stale `custom.py` / `CustomAgent` reference.
- **Test** `tests/unit/test_agent_loop.py` (resolver), `tests/unit/test_run_agentic_search.py` (CLI), and the web test path for app dispatch.

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
# tests/unit/test_agent_loop.py
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
# The four AgentLoopBase loops the registry consolidates. AgenticRAGLoop and the
# retrieval-pipeline modes are deliberately NOT here (different contract).
CANONICAL_AGENT_NAMES: frozenset[str] = frozenset(
    {"plain_generation", "single_turn_agent", "search_agent", "tool_agent"}
)

# CLI/web aliases → canonical registry name. Canonical names map to themselves.
_AGENT_ALIASES: dict[str, str] = {
    "single": "plain_generation",   # CLI --mode single = PlainGenerationLoop (preserve today's behavior)
    "search": "search_agent",
    "tool": "tool_agent",
    "plain_generation": "plain_generation",
    "single_turn_agent": "single_turn_agent",
    "search_agent": "search_agent",
    "tool_agent": "tool_agent",
}


def resolve_agent_name(name: str) -> str:
    """Resolve a CLI/web alias or canonical name to a canonical registry loop name.

    Raises KeyError for names that are not registry loops (e.g. chat_loop,
    search_tool, hybrid_search, chat_once) so callers keep dispatching those on
    their existing non-registry paths.
    """
    try:
        return _AGENT_ALIASES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown agent loop alias: {name!r}. Known: {sorted(_AGENT_ALIASES)}"
        ) from exc
```

In `src/__init__.py`, next to the existing registry exports (~`:17`):

```python
from .agents.base import resolve_agent_name as resolve_agent_name
from .agents.base import CANONICAL_AGENT_NAMES as CANONICAL_AGENT_NAMES
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_agent_loop.py -k "resolve or canonical_names_are_registered" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/base.py src/__init__.py tests/unit/test_agent_loop.py
git commit -m "feat: canonical agent names + resolve_agent_name alias resolver"
```

---

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

In each of `run_single_turn`/`run_search_agent`/`run_tool_agent` (`examples/run_agentic_search.py`), replace the hard-coded loop **class** reference with a registry lookup while keeping the loop-specific construction. Example for `run_search_agent` (~`:852`):

```python
    from src import get_registered_agent_loop, resolve_agent_name
    from src.agents.search import SearchAgentLoopConfig  # config stays per-loop
    loop_cls = get_registered_agent_loop(resolve_agent_name("search"))
    loop = loop_cls(
        tokenizer=tokenizer,
        server_manager=server_manager,
        search_config=SearchAgentLoopConfig(...),  # unchanged kwargs
    )
```

Apply the analogous change to `run_single_turn` (`resolve_agent_name("single")`) and `run_tool_agent` (`resolve_agent_name("tool")`). The `if args.mode == "single"/"search"/"tool"` dispatch block (~`:1236`) stays as-is (it routes to the `run_*` helpers); the registry lookup lives inside each helper. Do not touch any other mode.

- [ ] **Step 4: Run test + the existing CLI suite**

Run: `pytest tests/unit/test_run_agentic_search.py -v`
Expected: PASS (new test + all existing).

- [ ] **Step 5: Commit**

```bash
git add examples/run_agentic_search.py tests/unit/test_run_agentic_search.py
git commit -m "feat: CLI selects loop class via the registry"
```

---

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

In `src/internal/servers/web/app.py`, the `if mode == "search_agent":` block (~`:934`): replace `from src.agents.search import SearchAgentLoop, SearchAgentLoopConfig` + `loop = SearchAgentLoop(...)` with a registry class lookup, keeping the `SearchAgentLoopConfig` import and the construction kwargs:

```python
                from src import get_registered_agent_loop, resolve_agent_name
                from src.agents.search import SearchAgentLoopConfig
                # ... existing 400-guard unchanged ...
                loop_cls = get_registered_agent_loop(resolve_agent_name("search_agent"))
                loop = loop_cls(
                    tokenizer=tokenizer,
                    server_manager=manager,
                    search_config=SearchAgentLoopConfig(...),  # unchanged
                )
```

Apply the analogous change to the `tool_agent` block (~`:996`): `loop_cls = get_registered_agent_loop(resolve_agent_name("tool_agent"))`, keep `tools=` + `config=ToolAgentLoopConfig(...)`. **Do not touch** the `chat_loop` (AgenticRAGLoop), `search_tool`, `hybrid_search`, or `chat_once` blocks.

- [ ] **Step 4: Run the web test path**

Run: `pytest tests/unit/servers/web/ -v`
Expected: PASS (new + existing).

- [ ] **Step 5: Commit**

```bash
git add src/internal/servers/web/app.py tests/unit/servers/web/
git commit -m "feat: web search_agent/tool_agent select class via the registry"
```

---

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

Add a short prose section: the registry (`get_registered_agent_loop` + `resolve_agent_name`) is the single source of truth for the four `AgentLoopBase` loops; non-registry rows keep their own dispatch; construction is per-loop.

- [ ] **Step 2: Fix the stale CLAUDE.md reference**

In `.claude/CLAUDE.md`, find the line referencing `custom.py` — `CustomAgent` with configurable instructions + knowledge + tools` — and remove/replace it (the file does not exist). Replace with a pointer to `docs/agent-invocation-surface.md` and the registry.

Run: `grep -n "custom.py\|CustomAgent" .claude/CLAUDE.md`
Expected: no matches after the edit.

- [ ] **Step 3: Commit**

```bash
git add docs/agent-invocation-surface.md .claude/CLAUDE.md
git commit -m "docs: agent invocation surface table; fix stale custom.py reference"
```

---

## Self-Review

**Spec coverage:** registry as source of truth (Tasks 2,3 route class selection through it) · canonical names + alias resolver (Task 1) · `AgenticRAGLoop` non-registry (documented, Task 4; not registered per resolved open item) · scenario mapping (Task 4, doc not dead code — non-goal forbids reworking auto-detect) · documentation + custom.py fix (Task 4). The spec's "thin scenario→agent code table" is intentionally delivered as documentation; called out in Task 4 and the Global Constraints, consistent with the spec's non-goal of not reworking web auto-detection.

**Placeholder scan:** Tasks 2/3 test bodies adapt to existing harnesses (`test_run_agentic_search.py`, the web test dir) — the implementer reads those first; the asserted resolver mappings are concrete. Construction `...` ellipses in Tasks 2/3 mean "keep the existing kwargs unchanged" (explicitly stated), not omitted detail — the implementer preserves the current constructor args verbatim.

**Type consistency:** `resolve_agent_name(str) -> str`, `CANONICAL_AGENT_NAMES: frozenset[str]`, `get_registered_agent_loop(str) -> type[AgentLoopBase]` used identically across Tasks 1–3.
