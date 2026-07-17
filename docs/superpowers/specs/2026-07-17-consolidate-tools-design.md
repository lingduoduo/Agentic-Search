# Consolidate Tools + Dashboard Visibility — Design

**Date:** 2026-07-17
**Branch:** `feat/consolidate-tools`
**Status:** Approved (brainstorming)

## Motivation

The `semantic_router.py` + `knowledge_base.py` work (#422–#424) made the global
`ToolRegistry` the single source of truth for the web/agent process's tools: it
is seeded at startup, the dashboard lists it, and the router discovers over it.
This change finishes that consolidation:

1. **Complete the built-in set** — seed `rag_routing_tool` (currently skipped
   because the lifespan seeds `llm=None`), so the dashboard shows the full
   built-in toolset when an LLM is configured.
2. **Simplify** — remove the dead stub tool system in
   `src/internal/tools/built_in_tools.py`, which is a non-functional Onyx-era
   remnant (empty tool classes + an uncalled runner) that is not registered, not
   in the dashboard, and confuses the tool landscape.
3. **Confirm** the dashboard lists the full registry set end-to-end.

The 6 MCP-native tools stay a **separate FastMCP surface** (decision on record):
three are bound to the MCP request auth context and cannot run in the web
process. The existing `dynamic.py` bridge already exposes registry tools over
MCP (registry → MCP). Documented, not merged into the registry.

## Scope

### In scope
- Pass the resolved `llm` into the lifespan seed so `rag_routing_tool` registers
  when an LLM is configured (`src/internal/servers/web/app.py`).
- Surgically remove zero-importer dead code from
  `src/internal/tools/built_in_tools.py`.
- A short doc note (in the existing `docs/mcp.md` discovery section) that MCP
  tools remain a separate surface.
- Tests: extend `tests/unit/test_knowledge_base.py` to cover the LLM-present
  seeding path; a `built_in_tools` import/smoke check if warranted.

### Non-goals
- No merging MCP-native tools into the registry (auth-bound; separate surface).
- No removal of the still-imported `built_in_tools` symbols
  (`CITEABLE_TOOLS_NAMES`, `STOPPING_TOOLS_NAMES`, `TOOL_NAME_TO_CLASS`).
- No new tools; no new dependency; no `ToolAgentLoop` internal changes.

## Components

### 1. Seed `rag_routing_tool` — `src/internal/servers/web/app.py`

The lifespan currently seeds without the LLM (app.py:1284-1287):

```python
        seed_tools(
            tool_registry,
            tools=tool_knowledge_base(search_url=resolved.services.retrieval_url),
        )
```

Change to pass the `llm` already resolved in `create_web_app` scope
(app.py:1266-1279 builds `OpenAICompatibleLLM` when an API key is present, else
`None`):

```python
        seed_tools(
            tool_registry,
            tools=tool_knowledge_base(
                search_url=resolved.services.retrieval_url, llm=llm
            ),
        )
```

Effect: LLM configured → registry/dashboard gain `rag_routing_tool` (4 built-ins);
no LLM → unchanged (3). `tool_knowledge_base` already gates `rag_routing_tool` on
`llm is not None`, so this is safe with no LLM.

Note: `rag_routing_tool` has a unique name, so the #424 `_run_tool_agent`
`search`-dedup is unaffected; the tool-agent simply also gains it.

### 2. Simplify `src/internal/tools/built_in_tools.py`

Remove (all verified zero-importer):
- The 6 stub classes: `SearchTool`, `WebSearchTool`, `PythonTool`, `OpenURLTool`,
  `ImageGenerationTool`, `MemoryTool`.
- `run_tool_calls()` (never called) and `_ParallelToolCallResults` (only used by
  it).
- `extract_url_snippet_map()` (never imported).

Keep (still imported by `admin_surface.py` + `tool_call_args_streaming.py`):
- `CITEABLE_TOOLS_NAMES`, `STOPPING_TOOLS_NAMES`, `TOOL_NAME_TO_CLASS`.

Update the module docstring to reflect that it now holds only the tool-name sets
and the (currently empty) name→class placeholder. Drop the now-unused
`dataclass`/`field` import.

Resulting file is ~3 constants + docstring — no behavior change (the removed
symbols were dead).

### 3. Doc note — `docs/mcp.md`

The "Semantic tool discovery" section (rewritten in #424) already states MCP-native
tools stay FastMCP-only and the bridge mirrors registry → MCP. Add one sentence
noting the registry now also seeds `rag_routing_tool` when an LLM is configured,
so the built-in set is `web_search`, `search`, `search_routing_tool`
(+`rag_routing_tool` with an LLM). No structural change.

## Tests

- `tests/unit/test_knowledge_base.py` already covers `tool_knowledge_base(llm=...)`
  adding `rag_routing_tool` and `seed_tools` with an explicit 4-tool list. Add an
  assertion that `seed_tools(reg, tools=tool_knowledge_base(llm=object()))`
  registers `rag_routing_tool` into a fresh registry (name retrievable).
- `built_in_tools`: after the removal, add/keep a smoke test that the three kept
  symbols still import and that `run_tool_calls`/the stub classes are gone
  (`hasattr` false) — guards against re-introduction and confirms consumers still
  import cleanly. Verify `admin_surface` and `tool_call_args_streaming` still
  import.
- Manual end-to-end: `GET /admin/tools` through the real lifespan (as done for
  #424) returns the built-in set; with an LLM configured it includes
  `rag_routing_tool`.

## Success criteria

1. With an LLM configured, `seed_tools` registers 4 built-ins incl.
   `rag_routing_tool`; without, 3 (unchanged).
2. `built_in_tools.py` contains only the 3 live symbols + docstring; no importer
   breaks (`admin_surface`, `tool_call_args_streaming` import clean).
3. `pytest tests/unit/test_knowledge_base.py` (+ any built_in_tools test) passes;
   `ruff` clean; app imports.
4. Dashboard `GET /admin/tools` lists the full registry set end-to-end.
