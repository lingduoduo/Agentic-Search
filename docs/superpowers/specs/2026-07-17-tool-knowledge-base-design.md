# Tool Knowledge Base — Single Source of Truth for Executable Tools — Design

**Date:** 2026-07-17
**Branch:** `feat/tool-knowledge-base`
**Status:** Approved (brainstorming)

## Motivation

The repo has three disconnected "tool" surfaces: the dashboard's global
`ToolRegistry` singleton (empty by default → ToolPanel shows "No tools
registered"), `ToolAgentLoop`'s own per-loop registry (tools passed in), and the
semantic router's description catalog (`default_tool_catalog()`, which describes
the *MCP-server* tool surface, a different process).

We make the **global `ToolRegistry` singleton the single source of truth** for
the executable tools in this (web/agent) process. `tool_knowledge_base()`
provides the **built-in seed set** that `seed_tools()` loads into the registry at
startup; OpenAPI tools registered at runtime (via `register_from_openapi`) land
in the same registry. The dashboard, the agent, and the semantic router all read
the registry, so discovery covers **built-ins ∪ OpenAPI tools**.

This **replaces** the MCP-surface catalog added in #422 (see "What this reverses"
below). The maintainer chose the "replace / one KB" path over "coexist", and
chose the registry (not a static built-in list) as the source of truth, with eyes
open.

## Scope

### In scope
- New module `src/tools/knowledge_base.py`: `tool_knowledge_base()` (built-in
  seed set, flat `list[Tool]`) and `seed_tools()`.
- Reimplement `semantic_router.default_tool_catalog()` to derive from the **live
  registry** via the existing `catalog_from_registry(tool_registry)`.
- Wire `seed_tools(tool_registry)` into the web app lifespan so the dashboard is
  populated at startup.
- Update `tests/unit/test_semantic_router.py` (the tests tied to the old
  MCP-surface catalog) and add `tests/unit/test_knowledge_base.py`.
- Rewrite the "Semantic tool discovery" section of `docs/mcp.md`.

### Non-goals
- No changes to `ToolAgentLoop` internals (it can *optionally* be fed
  `tool_knowledge_base(...)` or `tool_registry.list_tools()`, but wiring the
  CLI/agent call sites is out of scope).
- No new dependency.
- No MCP-server-process seeding (separate process; out of scope — noted only).

### What this reverses from #422
1. `default_tool_catalog()` no longer returns the hardcoded MCP surface
   (`web_search`/`knowledge_base`/`answer` with `search_web`, `open_urls`, …). It
   is derived from the **live registry** via `catalog_from_registry(tool_registry)`.
2. The docs/mcp.md **drift-guard test** (`test_catalog_mcp_tools_match_docs_mcp_table`)
   and its `_documented_mcp_tools` helper are **deleted** — the catalog no longer
   equals the MCP tool table.
3. The docs/mcp.md "Semantic tool discovery" section is rewritten to describe the
   registry-as-source-of-truth model rather than the MCP surface.
4. `catalog_from_registry()` (from #422) becomes the router's default source, and
   `get_all_tools()` (#423) stays.

## Known consequences (accepted)

1. **Registry-backed, so stateful.** `default_tool_catalog()` now reflects
   whatever is registered, so it is **empty until `seed_tools()` runs** (or a tool
   is registered). The registry, not a static list, is the truth. The web app
   seeds at startup; a bare `discover_tools()` before seeding returns `[]`.
2. **Server grouping comes from the registry.** `catalog_from_registry()` groups
   built-in (`source="function"`) tools into a single `local` server and each
   OpenAPI provider (`provider_id`) into its own server. So with only built-ins
   seeded, the router sees a **single `local` server** (two-stage routing
   degenerates to one server) — but it gains a server per OpenAPI provider as
   providers register, which is exactly the intended "covers all Tool Registry"
   behavior.

## Tool-system landscape (why MCP tools stay separate)

The repo has four parallel tool systems. This design makes the `ToolRegistry` the
source of truth for the **web/agent process** only; the MCP-native tools are
intentionally left out because they cannot run outside the MCP process.

| System | Registered via | In `ToolRegistry`? |
|---|---|---|
| `src/tools/` executable Tools (`web_search`, `search`, `search_routing_tool`, `rag_routing_tool`) | `seed_tools()` (this design) + `@tool_registry.tool` | **Yes** (seeded) |
| OpenAPI tools | `register_from_openapi()` (dashboard, runtime) | **Yes** |
| `src/internal/mcp_server/tools/` (`search_web`, `open_urls`, `ask_agentic_search`, `retrieve_documents`, `expand_query`, `search_indexed_documents`) | `@mcp_server.tool()` (FastMCP) | **No** — FastMCP-only |
| `src/internal/tools/built_in_tools.py` (chat-path classes) | none central | **No** |

**Why the MCP-native tools are not pulled in:** three of them
(`search_indexed_documents`, `retrieve_documents`, `ask_agentic_search`) call
`authenticated_retrieve`, which needs the MCP request's bearer token — they only
work inside the MCP server process. Wrapping them as web-process `ToolRegistry`
tools would fail on invoke and couple `src/tools` → `mcp_server`.

**The bridge already unifies the correct direction.**
`src/internal/mcp_server/tools/dynamic.py` mirrors **`ToolRegistry` → FastMCP**
(`_sync_all()` at import, plus `sync_tool_to_mcp(name)` at runtime). So the
`ToolRegistry` is the source of truth and the MCP server *consumes* it — seeding
the registry with built-ins makes them available over MCP too, **within a
process**. Because the registry is a per-process singleton, exposing the built-ins
over MCP requires seeding the **MCP process's** registry as well; that
cross-process seeding is a **follow-up, out of scope here** (this design seeds the
web process only).

## Components

### 1. `src/tools/knowledge_base.py`

The built-in seed set — a flat list of executable `Tool`s — plus the seeder. No
`ToolServer`/`catalog_from_knowledge_base`: server grouping is the registry's job.

```python
DEFAULT_SEARCH_URL = "http://localhost:8000/retrieve"


def tool_knowledge_base(
    *,
    search_url: str = DEFAULT_SEARCH_URL,
    top_k: int = 5,
    llm=None,
) -> list[Tool]:
    """The built-in executable tools that seed the registry."""
    tools: list[Tool] = [
        MultiQueryWebSearchTool(provider="retrieval", search_url=search_url, page_size=top_k),
        build_search_tool(provider="retrieval", search_url=search_url, page_size=top_k),
        build_search_routing_tool(search_url=search_url, top_k=top_k),
    ]
    if llm is not None:
        tools.append(build_rag_routing_tool(llm=llm, search_url=search_url, top_k=top_k))
    return tools


def seed_tools(registry: ToolRegistry, *, tools: list[Tool] | None = None) -> int:
    """Register the built-in tools into *registry*. Returns the count registered.

    Uses tool_knowledge_base() when tools is None. source defaults to "function"
    (register()'s default), so the dashboard lists them under "Built-in function
    tools" and catalog_from_registry() groups them into the ``local`` server.
    """
    tools = tool_knowledge_base() if tools is None else tools
    for t in tools:
        registry.register(t)
    return len(tools)
```

Tool names produced: `web_search`, `search`, `search_routing_tool`, and
`rag_routing_tool` (only when `llm` is supplied). This module imports the tool
builders from `src.tools.search` / `src.tools.routing_tools` and `ToolRegistry`
from `src.tools.registry`; it does **not** import `semantic_router`.

### 2. `src/tools/semantic_router.py`

- Replace the body of `default_tool_catalog()` so it derives from the **live
  registry**:

```python
def default_tool_catalog() -> list[ServerDefinition]:
    """The default routing catalog — the live tool registry (built-ins + OpenAPI)."""
    from .registry import tool_registry

    return catalog_from_registry(tool_registry)
```

  The `tool_registry` import is done inside the function to keep import order
  robust (and because the catalog must be read *at call time*, so it reflects
  runtime OpenAPI registrations). `catalog_from_registry` already lives in this
  module (#422).

- Everything else (`ToolDefinition`, `ServerDefinition`, `RoutingConfig`,
  `SemanticRouter`, `StructuredRequestParser`, `discover_tools`,
  `catalog_from_registry`, `get_all_tools`) is unchanged. The hardcoded
  MCP-surface catalog body is removed.

### 3. Web app wiring — `src/internal/servers/web/app.py`

In `lifespan` (next to `seed_db(db)`, [app.py:1279](src/internal/servers/web/app.py#L1279)):

```python
from src.tools.registry import tool_registry
from src.tools.knowledge_base import seed_tools
...
seed_db(db)
seed_tools(tool_registry)   # populate the dashboard's global registry at startup
```

Seeds with the default KB (`llm=None` → the three `search` tools). The dashboard's
`GET /admin/tools` then lists and can Test-invoke them. (Passing the app's LLM to
also seed `rag_routing_tool` is a possible follow-up; out of scope here.)

### 4. Tests

**New `tests/unit/test_knowledge_base.py`** (uses a **fresh `ToolRegistry()`**, never
the global singleton, to stay isolated):
- `tool_knowledge_base()` default (`llm=None`) → a flat `list[Tool]` of three
  tools named `web_search`, `search`, `search_routing_tool`; every entry is a
  `Tool` instance.
- With a fake `llm` object → appends `rag_routing_tool` (length 4).
- `seed_tools(fresh_registry)` returns `3`, and `fresh_registry.get("web_search")`
  is not None (registered + retrievable); the entries have `source == "function"`.
- `catalog_from_registry(fresh_registry)` after seeding → a single `local` server
  whose tools include the three built-in names (ties seeding to the router path).

**Update `tests/unit/test_semantic_router.py`:**
- Delete `test_catalog_mcp_tools_match_docs_mcp_table` and `_documented_mcp_tools`.
- Replace `test_default_catalog_has_three_named_servers_with_expected_tools` with
  a test that seeds a **fresh `ToolRegistry()`**, calls
  `catalog_from_registry(reg)`, and asserts the `local` server + built-in tool
  names. (Do **not** assert on `default_tool_catalog()` directly — it reads the
  global singleton and would be order-dependent.)
- Rewrite the routing-relevance tests (`test_router_ranks_web_tools_...`,
  `test_router_ranks_knowledge_base_...`) to build an **explicit multi-server
  custom catalog** inline and assert ranking over it (they no longer have MCP
  servers). Keep the two-field `server_hint` test as-is (already custom-catalog).
- Update `test_get_all_tools_flattens_every_server` (#423) to flatten an
  **explicit custom catalog** rather than `default_tool_catalog()`.
- Any remaining test that called `default_tool_catalog()`/`discover_tools()`
  without seeding is repointed to an explicit catalog, so the suite never depends
  on global registry state.

### 5. `docs/mcp.md`

Rewrite the "Semantic tool discovery (server-side)" section:
- Describe the **`ToolRegistry` as the single source of truth**: built-ins are
  loaded by `seed_tools(tool_registry)` at web startup (from
  `tool_knowledge_base()`), OpenAPI tools are added at runtime via
  `register_from_openapi`, and discovery (`discover_tools` /
  `default_tool_catalog()` = `catalog_from_registry(tool_registry)`) covers the
  union.
- Replace the old MCP-surface catalog table: built-in seed tools are `web_search`,
  `search`, `search_routing_tool` (and `rag_routing_tool` when an LLM is
  configured); the router groups built-ins into a `local` server and each OpenAPI
  provider into its own server.
- Remove the `browser_search`/`source="mcp"` framing and the "matches the table
  above" drift note. Keep the reconciliation that MCP client-side selection is
  unchanged.
- Note seeding is per-process; the MCP server process is separate and not seeded
  here.

## Success criteria

1. `from src.tools.knowledge_base import tool_knowledge_base, seed_tools` works;
   `tool_knowledge_base()` returns the built-in `Tool` list; `seed_tools(reg)`
   registers them and returns the count.
2. `default_tool_catalog()` returns `catalog_from_registry(tool_registry)` — no
   import cycle, and it reflects whatever is currently registered.
3. `pytest tests/unit/test_knowledge_base.py tests/unit/test_semantic_router.py`
   passes; the old drift-guard test is gone; no test depends on global registry
   state.
4. The web lifespan calls `seed_tools(tool_registry)`; a unit test confirms
   `seed_tools` registers tools into a fresh registry, and (post-seed)
   `catalog_from_registry` surfaces them under `local`.
5. `docs/mcp.md` describes the registry-as-source-of-truth model; `ruff` clean.
