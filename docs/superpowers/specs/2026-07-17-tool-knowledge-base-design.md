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

We introduce **`tool_knowledge_base()`** as a single source of truth for the
**executable tools that run in this (web/agent) process**. The dashboard
registry, the agent, and the semantic router all derive from it, so there is one
place tools are defined.

This **replaces** the MCP-surface catalog added in #422 (see "What this reverses"
below). The maintainer chose the "replace / one KB" path over "coexist" with eyes
open.

## Scope

### In scope
- New module `src/tools/knowledge_base.py`: `ToolServer`, `tool_knowledge_base()`,
  `get_all_tools_kb()`, `catalog_from_knowledge_base()`, `seed_tools()`.
- Reimplement `semantic_router.default_tool_catalog()` to derive from the KB.
- Wire `seed_tools(tool_registry)` into the web app lifespan so the dashboard is
  populated at startup.
- Update `tests/unit/test_semantic_router.py` (the tests tied to the old
  MCP-surface catalog) and add `tests/unit/test_knowledge_base.py`.
- Rewrite the "Semantic tool discovery" section of `docs/mcp.md`.

### Non-goals
- No changes to `ToolAgentLoop` internals (it can *optionally* be fed
  `get_all_tools_kb(...)`, but wiring the CLI/agent call sites is out of scope).
- No new dependency.
- No MCP-server-process seeding (separate process; out of scope — noted only).

### What this reverses from #422
1. `default_tool_catalog()` no longer returns the hardcoded MCP surface
   (`web_search`/`knowledge_base`/`answer` with `search_web`, `open_urls`, …). It
   is derived from `tool_knowledge_base()` (the executable set).
2. The docs/mcp.md **drift-guard test** (`test_catalog_mcp_tools_match_docs_mcp_table`)
   and its `_documented_mcp_tools` helper are **deleted** — the catalog no longer
   equals the MCP tool table.
3. The docs/mcp.md "Semantic tool discovery" section is rewritten to describe the
   executable KB rather than the MCP surface.
4. `catalog_from_registry()` and `get_all_tools()` (from #422/#423) stay.

## Known consequence (accepted)

The executable tools that exist in `src/tools` are retrieval/search oriented, so
the default KB (`llm=None`) has a **single `search` server**. The semantic
router's two-stage server→tool ranking therefore degenerates to single-server
until an `llm` is supplied (which adds the `answer` server). This is the honest
shape of the in-process tool set; the router still returns the search tools.

## Components

### 1. `src/tools/knowledge_base.py`

```python
DEFAULT_SEARCH_URL = "http://localhost:8000/retrieve"


@dataclass
class ToolServer:
    name: str
    description: str
    tools: list[Tool]   # executable Tool objects


def tool_knowledge_base(
    *,
    search_url: str = DEFAULT_SEARCH_URL,
    top_k: int = 5,
    llm=None,
) -> list[ToolServer]:
    """The single source of truth for this process's runnable tools."""
```

**Contents:**

| server | tools (executable) | constructor |
|---|---|---|
| `search` | `web_search` | `MultiQueryWebSearchTool(provider="retrieval", search_url=search_url, page_size=top_k)` |
| `search` | `search` | `build_search_tool(provider="retrieval", search_url=search_url, page_size=top_k)` |
| `search` | `search_routing_tool` | `build_search_routing_tool(search_url=search_url, top_k=top_k)` |
| `answer` | `rag_routing_tool` | `build_rag_routing_tool(llm=llm, search_url=search_url, top_k=top_k)` — **only when `llm is not None`** |

Server descriptions:
- `search`: "Search the web and the indexed corpus for relevant documents."
- `answer`: "Answer a question using retrieval-augmented generation over the corpus."

**Companion functions (same module):**

```python
def get_all_tools_kb(servers: list[ToolServer]) -> list[Tool]:
    """Flatten the KB to a single list of executable tools."""

def catalog_from_knowledge_base(servers: list[ToolServer]) -> list[ServerDefinition]:
    """Derive the semantic router's *description* catalog from the KB.

    Each ToolServer -> ServerDefinition(name, description,
      tools=[ToolDefinition(name=t.name, description=t.schema.description,
                            source="function", server=name) for t in server.tools])
    """

def seed_tools(registry: ToolRegistry, *, kb: list[ToolServer] | None = None) -> int:
    """Register every executable tool from the KB into *registry*.

    Uses tool_knowledge_base() when kb is None. Returns the number registered.
    """
```

`catalog_from_knowledge_base` imports `ServerDefinition`/`ToolDefinition` from
`semantic_router` (dependency direction: `knowledge_base` → `semantic_router`).

### 2. `src/tools/semantic_router.py`

- Replace the body of `default_tool_catalog()` with a **lazy-import** derivation
  (to avoid a module-level import cycle, since `knowledge_base` imports
  `semantic_router`):

```python
def default_tool_catalog() -> list[ServerDefinition]:
    """The default routing catalog, derived from the executable tool KB."""
    from .knowledge_base import catalog_from_knowledge_base, tool_knowledge_base

    return catalog_from_knowledge_base(tool_knowledge_base())
```

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

**New `tests/unit/test_knowledge_base.py`:**
- `tool_knowledge_base()` default (`llm=None`) → one `search` server holding three
  executable `Tool`s named `web_search`, `search`, `search_routing_tool`; every
  entry is a `Tool` instance.
- With a fake `llm` → adds an `answer` server with `rag_routing_tool`.
- `get_all_tools_kb()` flattens to the executable tools (count matches).
- `catalog_from_knowledge_base()` derives `ServerDefinition`s whose
  `ToolDefinition`s carry the right `name`/`server` and `source="function"`.
- `seed_tools(fresh_registry)` returns the tool count and
  `fresh_registry.get("web_search")` is not None (registered + retrievable).

**Update `tests/unit/test_semantic_router.py`:**
- Delete `test_catalog_mcp_tools_match_docs_mcp_table` and `_documented_mcp_tools`.
- Rewrite `test_default_catalog_has_three_named_servers_with_expected_tools` →
  assert the derived default catalog has a `search` server whose tools are
  `web_search`, `search`, `search_routing_tool` (and no `answer` server at
  `llm=None`).
- Rewrite the routing-relevance tests (`test_router_ranks_web_tools_...`,
  `test_router_ranks_knowledge_base_...`) to use an explicit multi-server custom
  catalog (they no longer have MCP servers to rank). Keep the two-field
  `server_hint` test as-is (it already uses a custom catalog).
- Update `test_get_all_tools_flattens_every_server` (from #423) to assert against
  the new derived default catalog's tool names.

### 5. `docs/mcp.md`

Rewrite the "Semantic tool discovery (server-side)" section:
- Describe `tool_knowledge_base()` as the single source of the process's runnable
  tools, feeding the dashboard registry (via `seed_tools()` at startup), the
  agent, and the router (`discover_tools`).
- Replace the old MCP-surface catalog table with the executable KB table
  (`search`: `web_search`/`search`/`search_routing_tool`; `answer`:
  `rag_routing_tool` when an LLM is configured).
- Remove the `browser_search`/`source="mcp"` framing and the "matches the table
  above" drift note. Keep the reconciliation that MCP client-side selection is
  unchanged.
- Note that seeding is per-process; the MCP server process is separate.

## Success criteria

1. `from src.tools.knowledge_base import tool_knowledge_base, seed_tools` works;
   `seed_tools(tool_registry)` populates the registry with the executable tools.
2. `default_tool_catalog()` returns the derived catalog (no import cycle).
3. `pytest tests/unit/test_knowledge_base.py tests/unit/test_semantic_router.py`
   passes; the old drift-guard test is gone.
4. The web lifespan calls `seed_tools(tool_registry)`; a unit test confirms
   `seed_tools` registers tools into a fresh registry.
5. `docs/mcp.md` describes the KB, not the MCP surface; `ruff` clean.
