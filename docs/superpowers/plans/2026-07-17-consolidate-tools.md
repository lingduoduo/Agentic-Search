# Consolidate Tools + Dashboard Visibility — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the tool consolidation — seed `rag_routing_tool` so the dashboard shows the full built-in set, and remove the dead stub tool system.

**Architecture:** Pass the lifespan's already-resolved `llm` into `tool_knowledge_base()` so `rag_routing_tool` is seeded when an LLM is configured. Surgically delete zero-importer dead code from `src/internal/tools/built_in_tools.py`. Add a one-sentence docs note. MCP-native tools stay a separate bridged surface (no code change).

**Tech Stack:** Python, `pytest`, `ruff`.

## Global Constraints

- `tool_knowledge_base()` already gates `rag_routing_tool` on `llm is not None`; passing `llm=None` keeps the current 3-tool behavior.
- Only remove `built_in_tools` symbols with **zero importers**: the 6 stub classes (`SearchTool`, `WebSearchTool`, `PythonTool`, `OpenURLTool`, `ImageGenerationTool`, `MemoryTool`), `run_tool_calls`, `_ParallelToolCallResults`, `extract_url_snippet_map`. **Keep** `CITEABLE_TOOLS_NAMES`, `STOPPING_TOOLS_NAMES`, `TOOL_NAME_TO_CLASS`.
- No MCP-tool registration; no new tools; no new dependency; no `ToolAgentLoop` internal change.
- `ruff` clean; app imports without error.

---

### Task 1: Seed `rag_routing_tool` in the web lifespan

**Files:**
- Modify: `src/internal/servers/web/app.py`

**Interfaces:**
- Consumes: `tool_knowledge_base`, `seed_tools` (already imported), `llm` (in `create_web_app` scope), `resolved.services.retrieval_url`.
- Produces: registry seeded with `rag_routing_tool` when `llm` is present.

- [ ] **Step 1: Make the change**

In `src/internal/servers/web/app.py`, the lifespan currently seeds without the
LLM (lines 1284-1287):

```python
        seed_tools(
            tool_registry,
            tools=tool_knowledge_base(search_url=resolved.services.retrieval_url),
        )
```

Change to pass the resolved `llm`:

```python
        seed_tools(
            tool_registry,
            tools=tool_knowledge_base(
                search_url=resolved.services.retrieval_url, llm=llm
            ),
        )
```

`llm` is already resolved earlier in `create_web_app` (app.py:1266-1279: built
from config when an API key is present, else `None`) and is in the lifespan's
closure scope.

- [ ] **Step 2: Verify import + seeding path**

The seeding-with-LLM behavior is already unit-tested in
`tests/unit/test_knowledge_base.py::test_seed_tools_accepts_explicit_tools`
(`seed_tools(reg, tools=tool_knowledge_base(llm=object())) == 4` and
`reg.get("rag_routing_tool") is not None`). Confirm the app change:

```bash
python3 -c "import src.internal.servers.web.app as a; print('import ok')"
grep -n "search_url=resolved.services.retrieval_url, llm=llm" src/internal/servers/web/app.py
python3 -m pytest tests/unit/test_knowledge_base.py -q
ruff check src/internal/servers/web/app.py
```
Expected: `import ok`; grep shows the change; tests pass; ruff clean.

- [ ] **Step 3: Commit**

```bash
git add src/internal/servers/web/app.py
git commit -m "feat: seed rag_routing_tool when an LLM is configured"
```

---

### Task 2: Remove the dead stub system from built_in_tools.py

**Files:**
- Modify: `src/internal/tools/built_in_tools.py`
- Test: `tests/unit/test_built_in_tools.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `built_in_tools` exposing only `CITEABLE_TOOLS_NAMES`, `STOPPING_TOOLS_NAMES`, `TOOL_NAME_TO_CLASS`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_built_in_tools.py`:

```python
from __future__ import annotations

import src.internal.tools.built_in_tools as bit


def test_live_symbols_present():
    assert bit.CITEABLE_TOOLS_NAMES == {"search", "web_search", "open_url"}
    assert bit.STOPPING_TOOLS_NAMES == {"image_generation"}
    assert bit.TOOL_NAME_TO_CLASS == {}


def test_dead_stub_symbols_removed():
    for name in (
        "SearchTool",
        "WebSearchTool",
        "PythonTool",
        "OpenURLTool",
        "ImageGenerationTool",
        "MemoryTool",
        "run_tool_calls",
        "extract_url_snippet_map",
        "_ParallelToolCallResults",
    ):
        assert not hasattr(bit, name), f"{name} should be removed"


def test_consumers_still_import():
    import src.internal.chat.tool_call_args_streaming  # noqa: F401
    import src.internal.observability.admin_surface  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_built_in_tools.py -v`
Expected: `test_dead_stub_symbols_removed` FAILS (the stub symbols still exist).

- [ ] **Step 3: Rewrite the module**

Replace the entire contents of `src/internal/tools/built_in_tools.py` with:

```python
"""Built-in tool name sets used by the chat and observability surfaces.

These constants classify tool behavior (citeable / stopping) for the
observability admin surface and chat streaming. ``TOOL_NAME_TO_CLASS`` is a
placeholder name→class map, currently empty (no built-in tool classes are
registered through it in this repo).
"""

from __future__ import annotations

# Tool names that produce citable documents (trigger citation reminders)
CITEABLE_TOOLS_NAMES: set[str] = {"search", "web_search", "open_url"}

# Tool names that stop the loop after running (e.g. image generation)
STOPPING_TOOLS_NAMES: set[str] = {"image_generation"}

# Placeholder name→class map (empty; no built-in tool classes are registered)
TOOL_NAME_TO_CLASS: dict[str, type] = {}
```

This drops the 6 stub classes, `run_tool_calls`, `_ParallelToolCallResults`,
`extract_url_snippet_map`, and the now-unused `dataclass`/`field` import.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_built_in_tools.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Confirm no other consumer broke + lint**

Run:
```bash
grep -rnE "run_tool_calls|extract_url_snippet_map|_ParallelToolCallResults|import SearchTool|WebSearchTool|PythonTool|OpenURLTool|ImageGenerationTool|MemoryTool" src --include='*.py' | grep -v __pycache__ | grep -viE "built_in_tools|test_built_in_tools|agents/components/search_tool|MultiQueryWebSearchTool|SearchToolCall|SearchToolUsage|SearchToolStart|SearchToolQueries|SearchToolDocuments" || echo "no dead-symbol consumers"
ruff check src/internal/tools/built_in_tools.py tests/unit/test_built_in_tools.py
```
Expected: `no dead-symbol consumers`; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/internal/tools/built_in_tools.py tests/unit/test_built_in_tools.py
git commit -m "refactor: drop dead stub tool classes from built_in_tools"
```

---

### Task 3: Docs note

**Files:**
- Modify: `docs/mcp.md`

**Interfaces:**
- Consumes: nothing (docs only).

- [ ] **Step 1: Add the note**

In `docs/mcp.md`, in the "Semantic tool discovery (server-side)" section, find
the sentence listing the built-in seed tools (`web_search`, `search`,
`search_routing_tool` ...) and ensure it reads that `rag_routing_tool` is
included **when an LLM is configured** — i.e. the built-in set is `web_search`,
`search`, `search_routing_tool` (+ `rag_routing_tool` when an LLM is configured).
If the section already says "(and `rag_routing_tool` when an LLM is configured)",
leave it; otherwise adjust that one clause. Do not restructure the section.

- [ ] **Step 2: Verify**

Run:
```bash
grep -n "rag_routing_tool" docs/mcp.md
python3 -c "import pathlib; d=pathlib.Path('docs/mcp.md').read_text(); assert 'rag_routing_tool' in d and 'when an LLM is configured' in d; print('ok')"
```
Expected: the clause is present; `ok`.

- [ ] **Step 3: Commit**

```bash
git add docs/mcp.md
git commit -m "docs: note rag_routing_tool is seeded when an LLM is configured"
```

---

## Self-Review

**Spec coverage:**
- Seed `rag_routing_tool` via lifespan `llm` → Task 1. ✓
- Surgical removal of dead `built_in_tools` symbols; keep the 3 live ones → Task 2. ✓
- Docs note that MCP stays separate + rag seeding → Task 3. ✓
- Non-goals (no MCP registration, no new tools/dep, keep the 3 imported symbols) → enforced by Global Constraints. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. Task 3 is conditional but gives the exact target clause.

**Type consistency:** `tool_knowledge_base(search_url=..., llm=llm)` matches its signature; the kept `built_in_tools` symbols keep their exact types (`set[str]`, `dict[str, type]`); test references match the removed/kept names. ✓
