# Tool Knowledge Base — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `ToolRegistry` the single source of truth for the web/agent process's executable tools: a `tool_knowledge_base()` built-in seed set loaded at web startup, a registry-derived router catalog, and a rewritten docs section.

**Architecture:** New `src/tools/knowledge_base.py` builds the built-in `Tool`s and `seed_tools()` registers them into the global `tool_registry`. `semantic_router.default_tool_catalog()` is reimplemented as `catalog_from_registry(tool_registry)` so discovery covers built-ins (seeded) + OpenAPI tools. The web lifespan calls `seed_tools()`. MCP-native tools stay FastMCP-only.

**Tech Stack:** Python, `scikit-learn`/`numpy` (already deps), `pytest`, `ruff`.

## Global Constraints

- `tool_knowledge_base(*, search_url="http://localhost:8000/retrieve", top_k=5, llm=None) -> list[Tool]` returns built-in tools named `web_search`, `search`, `search_routing_tool` (+ `rag_routing_tool` only when `llm is not None`).
- `seed_tools(registry, *, tools=None) -> int` registers each tool (default `source="function"`) and returns the count.
- `default_tool_catalog()` must be `catalog_from_registry(tool_registry)` read at call time (lazy import of `tool_registry`); no import cycle.
- No new dependency; no changes to `ToolAgentLoop` internals or the MCP server.
- Tests must not depend on the global `tool_registry` state — use a fresh `ToolRegistry()` or explicit catalogs.
- `ruff check` / `ruff format` clean.
- Default seeding uses `llm=None` (3 tools). MCP-native tools are NOT registered (auth-bound; out of scope).

---

### Task 1: `knowledge_base.py` — built-in tools + seeder

**Files:**
- Create: `src/tools/knowledge_base.py`
- Test: `tests/unit/test_knowledge_base.py`

**Interfaces:**
- Consumes: `MultiQueryWebSearchTool`, `build_search_tool` (`src/tools/search.py`); `build_search_routing_tool`, `build_rag_routing_tool` (`src/tools/routing_tools.py`); `ToolRegistry` (`src/tools/registry.py`); `Tool` (`src/tools/base.py`); `catalog_from_registry` (`src/tools/semantic_router.py`).
- Produces: `tool_knowledge_base(*, search_url=..., top_k=5, llm=None) -> list[Tool]`; `seed_tools(registry, *, tools=None) -> int`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_knowledge_base.py`:

```python
from __future__ import annotations

from src.tools.base import Tool
from src.tools.knowledge_base import seed_tools, tool_knowledge_base
from src.tools.registry import ToolRegistry
from src.tools.semantic_router import catalog_from_registry

BUILTIN_NAMES = {"web_search", "search", "search_routing_tool"}


def test_knowledge_base_default_has_three_builtin_tools():
    tools = tool_knowledge_base()
    assert all(isinstance(t, Tool) for t in tools)
    assert {t.name for t in tools} == BUILTIN_NAMES


def test_knowledge_base_adds_rag_tool_when_llm_present():
    tools = tool_knowledge_base(llm=object())
    assert {t.name for t in tools} == BUILTIN_NAMES | {"rag_routing_tool"}


def test_seed_tools_registers_into_fresh_registry():
    reg = ToolRegistry()
    count = seed_tools(reg)
    assert count == 3
    assert reg.get("web_search") is not None
    # Built-ins register under source="function".
    assert all(e.source == "function" for e in reg.list())


def test_seeded_registry_surfaces_in_catalog_from_registry():
    reg = ToolRegistry()
    seed_tools(reg)
    catalog = catalog_from_registry(reg)
    # Function tools group into a single "local" server.
    by_name = {s.name: s for s in catalog}
    assert set(by_name) == {"local"}
    assert {t.name for t in by_name["local"].tools} == BUILTIN_NAMES


def test_seed_tools_accepts_explicit_tools():
    reg = ToolRegistry()
    tools = tool_knowledge_base(llm=object())
    assert seed_tools(reg, tools=tools) == 4
    assert reg.get("rag_routing_tool") is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_knowledge_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.tools.knowledge_base'`

- [ ] **Step 3: Write minimal implementation**

Create `src/tools/knowledge_base.py`:

```python
"""Built-in executable tools that seed the global ToolRegistry.

The ToolRegistry is the single source of truth for this process's runnable
tools. ``tool_knowledge_base()`` is the built-in seed set; ``seed_tools()``
registers it. OpenAPI tools are added separately at runtime via
``register_from_openapi``. MCP-native tools live in the MCP server process and
are not registered here.
"""

from __future__ import annotations

from .base import Tool
from .registry import ToolRegistry
from .routing_tools import build_rag_routing_tool, build_search_routing_tool
from .search import MultiQueryWebSearchTool, build_search_tool

DEFAULT_SEARCH_URL = "http://localhost:8000/retrieve"


def tool_knowledge_base(
    *,
    search_url: str = DEFAULT_SEARCH_URL,
    top_k: int = 5,
    llm=None,
) -> list[Tool]:
    """The built-in executable tools that seed the registry.

    ``rag_routing_tool`` is included only when an ``llm`` is supplied (it needs
    a live LLM client).
    """
    tools: list[Tool] = [
        MultiQueryWebSearchTool(
            provider="retrieval", search_url=search_url, page_size=top_k
        ),
        build_search_tool(
            provider="retrieval", search_url=search_url, page_size=top_k
        ),
        build_search_routing_tool(search_url=search_url, top_k=top_k),
    ]
    if llm is not None:
        tools.append(
            build_rag_routing_tool(llm=llm, search_url=search_url, top_k=top_k)
        )
    return tools


def seed_tools(registry: ToolRegistry, *, tools: list[Tool] | None = None) -> int:
    """Register the built-in tools into *registry*; return the count.

    Uses ``tool_knowledge_base()`` when *tools* is None. Tools register with the
    default ``source="function"``, so the dashboard lists them under "Built-in
    function tools" and ``catalog_from_registry`` groups them into ``local``.
    """
    tools = tool_knowledge_base() if tools is None else tools
    for tool in tools:
        registry.register(tool)
    return len(tools)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_knowledge_base.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tools/knowledge_base.py tests/unit/test_knowledge_base.py
git commit -m "feat: add tool_knowledge_base built-in seed set and seed_tools"
```

---

### Task 2: Registry-derived `default_tool_catalog()` + router test rework

**Files:**
- Modify: `src/tools/semantic_router.py`
- Modify: `tests/unit/test_semantic_router.py`

**Interfaces:**
- Consumes: `catalog_from_registry` (same module), `tool_registry` (`src/tools/registry.py`).
- Produces: `default_tool_catalog() -> list[ServerDefinition]` returning `catalog_from_registry(tool_registry)`.

- [ ] **Step 1: Repoint the tests (they will fail until the impl changes)**

In `tests/unit/test_semantic_router.py`:

**(a) Delete** these three, entirely:
- `test_default_catalog_has_three_named_servers_with_expected_tools`
- `_documented_mcp_tools`
- `test_catalog_mcp_tools_match_docs_mcp_table`

**(b) Add** a shared fixture near the top (after the imports) — this is the
old MCP-surface catalog, now used only as an explicit test catalog:

```python
def _sample_catalog() -> list[ServerDefinition]:
    """A fixed multi-server catalog for router-ranking tests (not production)."""
    return [
        ServerDefinition(
            "web_search",
            "Search the public internet for news and fetch page content from URLs.",
            [
                ToolDefinition("search_web", "Search the public internet.", "mcp", "web_search"),
                ToolDefinition("open_urls", "Fetch full text of web page URLs.", "mcp", "web_search"),
                ToolDefinition("browser_search", "Browser-driven web search.", "retrieval-server", "web_search"),
            ],
        ),
        ServerDefinition(
            "knowledge_base",
            "Search and retrieve documents from the private indexed corpus.",
            [
                ToolDefinition("search_indexed_documents", "Search the private knowledge base.", "mcp", "knowledge_base"),
                ToolDefinition("retrieve_documents", "Retrieve raw indexed document content.", "mcp", "knowledge_base"),
                ToolDefinition("expand_query", "Expand a query into keyword variants.", "mcp", "knowledge_base"),
            ],
        ),
        ServerDefinition(
            "answer",
            "Synthesize a grounded answer from retrieved evidence.",
            [
                ToolDefinition("ask_agentic_search", "Synthesize a cited answer.", "mcp", "answer"),
                ToolDefinition("rag_routing_tool", "Answer via retrieval-augmented generation.", "function", "answer"),
            ],
        ),
    ]
```

**(c) Repoint** every remaining `default_tool_catalog()` call to `_sample_catalog()`:
- `test_router_ranks_web_tools_for_internet_request`: `SemanticRouter(_sample_catalog())`
- `test_router_ranks_knowledge_base_for_internal_docs_request`: `SemanticRouter(_sample_catalog())`
- `test_threshold_filters_zero_similarity_requests`: `SemanticRouter(_sample_catalog(), RoutingConfig(similarity_threshold=0.5))`
- `test_top_k_larger_than_catalog_is_clamped_and_deduped`: `SemanticRouter(_sample_catalog(), RoutingConfig(top_k_servers=10, top_k_tools=10))`
- `test_routing_details_shape`: `SemanticRouter(_sample_catalog())`
- `test_get_all_tools_flattens_every_server`: `catalog = _sample_catalog()` (the 8-name assertion stays valid)

**(d) Repoint** the two `discover_tools` tests to pass the explicit catalog:
- `test_discover_tools_unstructured_web_request`:
  `tools = discover_tools("search the public internet for recent news", catalog=_sample_catalog())`
- `test_discover_tools_uses_structured_server_hint`:
  `tools = discover_tools(request, catalog=_sample_catalog())`

- [ ] **Step 2: Run the repointed tests — they still pass on old impl, but the deleted-test file must import cleanly**

Run: `pytest tests/unit/test_semantic_router.py -q`
Expected: PASS (the repointed tests use `_sample_catalog()`, which equals the current `default_tool_catalog()` output, so they pass before the impl change too). If any fail, fix the repoint before continuing.

- [ ] **Step 3: Change `default_tool_catalog()` to derive from the registry**

In `src/tools/semantic_router.py`, replace the entire `default_tool_catalog`
function body (lines 34-110, the hardcoded three-server catalog) with:

```python
def default_tool_catalog() -> list[ServerDefinition]:
    """The default routing catalog: the live tool registry (built-ins + OpenAPI).

    Read at call time so it reflects tools registered at runtime (e.g. OpenAPI
    providers). Empty until the registry is seeded / a tool is registered.
    """
    from .registry import tool_registry

    return catalog_from_registry(tool_registry)
```

(`catalog_from_registry` is defined lower in the same module; calling it before
its definition is fine because `default_tool_catalog` only runs at call time.)

- [ ] **Step 4: Run the full router test file**

Run: `pytest tests/unit/test_semantic_router.py -v`
Expected: PASS. No test calls `default_tool_catalog()` anymore, so none depends
on global registry state.

- [ ] **Step 5: Commit**

```bash
git add src/tools/semantic_router.py tests/unit/test_semantic_router.py
git commit -m "feat: derive default_tool_catalog from the live registry; rework router tests"
```

---

### Task 3: Seed the registry from the web lifespan

**Files:**
- Modify: `src/internal/servers/web/app.py`

**Interfaces:**
- Consumes: `seed_tools` (`src/tools/knowledge_base.py`), `tool_registry` (`src/tools/registry.py`).
- Produces: the dashboard's `GET /admin/tools` returns the built-ins after startup.

- [ ] **Step 1: Add the imports**

In `src/internal/servers/web/app.py`, near the existing
`from src.internal.servers.web.seeding import seed_db` (line ~100), add:

```python
from src.tools.knowledge_base import seed_tools
from src.tools.registry import tool_registry
```

- [ ] **Step 2: Call `seed_tools` in the lifespan**

In the `lifespan` async context manager, immediately after `seed_db(db)`
(line ~1279), add:

```python
        seed_tools(tool_registry)
```

- [ ] **Step 3: Verify the module imports and the call is wired**

Run:
```bash
python3 -c "import src.internal.servers.web.app as a; print('import ok')"
grep -n "seed_tools(tool_registry)" src/internal/servers/web/app.py
```
Expected: prints `import ok`; grep shows the call inside the lifespan.

(Note: full lifespan/dashboard verification requires booting the web app, which
loads `SEARCH_AGENT_MODEL` and is slow/hangs in CI — see
`examples/run_web_integration_tests.sh`. The `seed_tools` behavior itself is
unit-tested against a fresh registry in Task 1; this task only wires the call.)

- [ ] **Step 4: Commit**

```bash
git add src/internal/servers/web/app.py
git commit -m "feat: seed built-in tools into the registry at web startup"
```

---

### Task 4: Rewrite the docs/mcp.md "Semantic tool discovery" section

**Files:**
- Modify: `docs/mcp.md`

**Interfaces:**
- Consumes: nothing (docs only).

- [ ] **Step 1: Replace the section body**

In `docs/mcp.md`, replace the entire existing "## Semantic tool discovery
(server-side)" section (from that heading up to, but not including, the next
`##` heading) with:

```markdown
## Semantic tool discovery (server-side)

The **`ToolRegistry` is the single source of truth** for the web/agent process's
runnable tools. `src/tools/knowledge_base.py` provides the built-in seed set, and
`seed_tools(tool_registry)` loads it into the registry at web startup; OpenAPI
tools are added at runtime via `register_from_openapi`. Discovery covers the
union of both:

- `discover_tools(request)` returns the tools most relevant to a natural-language
  request, using a two-stage TF-IDF match (rank servers, then tools).
- `default_tool_catalog()` is `catalog_from_registry(tool_registry)`, read at call
  time — so it reflects whatever is registered (empty until seeding runs). Built-in
  tools group into a `local` server; each OpenAPI provider gets its own server.

Built-in seed tools: `web_search`, `search`, `search_routing_tool` (and
`rag_routing_tool` when an LLM is configured).

This does not change how MCP clients invoke tools — MCP tool selection stays
client-driven, as described above. Discovery is a ranking aid, not a dispatcher.

**Relationship to the MCP tools:** the `@mcp_server.tool()` functions
(`search_web`, `open_urls`, `ask_agentic_search`, `retrieve_documents`,
`expand_query`, `search_indexed_documents`) are registered with FastMCP, not the
`ToolRegistry`, and several are bound to the MCP request's auth context. The
`dynamic.py` bridge mirrors the `ToolRegistry` **into** the MCP server
(`sync_tool_to_mcp`), so the registry feeds MCP — not the reverse. Seeding is
per-process; exposing the built-ins over MCP would require seeding the MCP
server process's own registry (a separate follow-up).
```

- [ ] **Step 2: Verify placement and links**

Run:
```bash
python3 -c "import pathlib; d=pathlib.Path('docs/mcp.md').read_text(); assert 'ToolRegistry` is the single source of truth' in d; assert d.index('Semantic tool discovery') < d.index('## Resources'); print('ok')"
grep -n "browser_search\|source=.mcp.\|drift" docs/mcp.md || echo "no stale MCP-surface framing"
```
Expected: `ok`; the old `browser_search`/`source="mcp"` framing is gone from this section.

- [ ] **Step 3: Commit**

```bash
git add docs/mcp.md
git commit -m "docs: rewrite semantic tool discovery for registry-as-source-of-truth"
```

---

## Self-Review

**Spec coverage:**
- `tool_knowledge_base()` + `seed_tools()` → Task 1. ✓
- `default_tool_catalog()` = `catalog_from_registry(tool_registry)` → Task 2. ✓
- Delete drift-guard test + `_documented_mcp_tools`; repoint router tests off the global registry → Task 2. ✓
- Web lifespan seeding → Task 3. ✓
- docs/mcp.md rewrite (registry source of truth, MCP relationship, bridge) → Task 4. ✓
- Non-goals (no MCP-tool registration, no ToolAgentLoop change, no new dep) → enforced by Global Constraints. ✓
- Open item 1 (LLM seeding): Task 3 seeds `llm=None` (3 tools), matching the spec default. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. Task 3's verification is explicitly limited (app boot is slow) with the reason stated.

**Type consistency:** `tool_knowledge_base(...) -> list[Tool]`, `seed_tools(registry, *, tools=None) -> int`, `default_tool_catalog() -> list[ServerDefinition]`, and `catalog_from_registry` usage are consistent across Tasks 1-3. `_sample_catalog()` returns `list[ServerDefinition]` matching `SemanticRouter`'s input. ✓
