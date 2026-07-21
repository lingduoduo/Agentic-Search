# Tool Registry Inspection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only Dev Console surface that lists registered tools, the discovery catalog grouped by server, and a query-driven semantic-discovery view.

**Architecture:** Two handlers appended to the existing `create_debug_router` factory (`/api/debug` prefix, auto-mounted by the `if debug_panels:` block — no new wiring, no auth). They read the process-wide `tool_registry` singleton and `src.tools.semantic_router`. A new React `ToolsPanel` under `web/src/components/debug/` fetches them and renders the catalog + discovery, wired into `DevConsole`.

**Tech Stack:** FastAPI + Pydantic v2 (2.11.7) backend; React 19 + Vite + TypeScript frontend; pytest + FastAPI TestClient.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-21-tool-registry-inspection-design.md`.
- Read-only: no tool invocation, no registry/semantic-router changes, no new public `/api/tools`.
- Backend handlers must never 500 on an empty registry — return empty lists.
- Follow existing debug-router style: handlers defined inside `create_debug_router`, module-singleton `tool_registry` imported locally in the handler.
- Frontend fetchers go through the shared `requestJson<T>` helper; panel follows the `EvalResultsPanel.tsx` `useEffect`+`alive` pattern.

---

### Task 1: `GET /api/debug/tools` — registered list + catalog

**Files:**
- Modify: `src/internal/servers/web/debug_router.py`
- Test: `tests/unit/servers/web/test_debug_tools.py` (create)

**Interfaces:**
- Consumes: `tool_registry.all_summaries()` → `list[dict]` with keys
  `name, description, parameters, source, provider_id`; `catalog_from_registry(tool_registry)` → `list[ServerDefinition]` (dataclass with `name, description, tools:[ToolDefinition{name,description,source,server}]`).
- Produces: `GET /api/debug/tools` → `{"registered": [...], "catalog": [...]}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/servers/web/test_debug_tools.py
from __future__ import annotations

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.servers.web.debug_router import create_debug_router
from src.tools.base import FunctionTool
from src.tools.registry import tool_registry


def _client() -> TestClient:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    app = FastAPI()
    app.include_router(create_debug_router(search_url="http://retrieval:8001/retrieve", http_client=http_client))
    return TestClient(app)


def test_tools_lists_registered_and_catalog():
    stub = FunctionTool(lambda query: "ok", name="stub_tool_dbg", description="a stub", parameters={})
    tool_registry.register(stub, source="function")
    try:
        resp = _client().get("/api/debug/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert "stub_tool_dbg" in {t["name"] for t in data["registered"]}
        local = next(s for s in data["catalog"] if s["name"] == "local")
        assert any(t["name"] == "stub_tool_dbg" for t in local["tools"])
    finally:
        tool_registry.unregister("stub_tool_dbg")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/servers/web/test_debug_tools.py::test_tools_lists_registered_and_catalog -v`
Expected: FAIL — 404 (route not defined).

- [ ] **Step 3: Add the handler**

In `src/internal/servers/web/debug_router.py`, inside `create_debug_router` (next to the other `@router.get` handlers), add:

```python
    @router.get("/tools")
    def tools() -> dict:
        """Registered tools plus the discovery catalog grouped by server.

        Reads the process-wide ``tool_registry`` singleton (seeded at web
        startup). Empty registry → empty lists, never 500.
        """
        from dataclasses import asdict

        from src.tools.registry import tool_registry
        from src.tools.semantic_router import catalog_from_registry

        return {
            "registered": tool_registry.all_summaries(),
            "catalog": [asdict(s) for s in catalog_from_registry(tool_registry)],
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/servers/web/test_debug_tools.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/servers/web/debug_router.py tests/unit/servers/web/test_debug_tools.py
git commit -m "feat(debug): GET /api/debug/tools lists registered tools + catalog"
```

---

### Task 2: `POST /api/debug/tools/discover` — semantic discovery

**Files:**
- Modify: `src/internal/servers/web/debug_router.py`
- Test: `tests/unit/servers/web/test_debug_tools.py` (extend)

**Interfaces:**
- Consumes: `SemanticRouter(default_tool_catalog()).get_routing_details(query)` → `dict` with keys `request, stage1_servers, stage2_tools, final_tools`.
- Produces: `POST /api/debug/tools/discover` body `{"query": str}` → that dict. Blank/whitespace query → 422.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/servers/web/test_debug_tools.py`:

```python
def test_tools_discover_ranks_relevant_tool():
    stub = FunctionTool(
        lambda query: "ok",
        name="wikipedia_search_dbg",
        description="Search Wikipedia for encyclopedia articles about a topic.",
        parameters={},
    )
    tool_registry.register(stub, source="function")
    try:
        resp = _client().post("/api/debug/tools/discover", json={"query": "search wikipedia articles"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["request"] == "search wikipedia articles"
        assert "stage1_servers" in data and "final_tools" in data
        assert any(t["name"] == "wikipedia_search_dbg" for t in data["final_tools"])
    finally:
        tool_registry.unregister("wikipedia_search_dbg")


def test_tools_discover_blank_query_is_422():
    resp = _client().post("/api/debug/tools/discover", json={"query": "   "})
    assert resp.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/servers/web/test_debug_tools.py -v -k discover`
Expected: FAIL — 404 (route not defined).

- [ ] **Step 3: Add the request model and handler**

In `debug_router.py`, extend the pydantic import and add the model near `DebugQueryTransformRequest`:

```python
from pydantic import BaseModel, Field, field_validator
```

```python
class DebugToolDiscoverRequest(BaseModel):
    query: str = Field(..., min_length=1)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v
```

Then inside `create_debug_router` add the handler:

```python
    @router.post("/tools/discover")
    def tools_discover(req: DebugToolDiscoverRequest) -> dict:
        """Rank tools for *query* via the semantic router (TF-IDF, no LLM)."""
        from src.tools.semantic_router import SemanticRouter, default_tool_catalog

        return SemanticRouter(default_tool_catalog()).get_routing_details(req.query)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/servers/web/test_debug_tools.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Commit**

```bash
git add src/internal/servers/web/debug_router.py tests/unit/servers/web/test_debug_tools.py
git commit -m "feat(debug): POST /api/debug/tools/discover semantic tool ranking"
```

---

### Task 3: Dev Console "Tools" panel

**Files:**
- Modify: `web/src/types.ts`
- Modify: `web/src/api.ts`
- Create: `web/src/components/debug/ToolsPanel.tsx`
- Modify: `web/src/components/debug/DevConsole.tsx`

**Interfaces:**
- Consumes: `GET /api/debug/tools` → `{registered: ToolView[], catalog: CatalogServer[]}`; `POST /api/debug/tools/discover` → `ToolDiscoverResult`.
- Produces: `<ToolsPanel/>` rendered inside `DevConsole`.

- [ ] **Step 1: Add types**

Append to `web/src/types.ts` (reusing the existing `ToolView` at line 183 for `registered`):

```typescript
export interface CatalogTool {
  name: string;
  description: string;
  source: string;
  server: string;
}

export interface CatalogServer {
  name: string;
  description: string;
  tools: CatalogTool[];
}

export interface DebugToolsResult {
  registered: ToolView[];
  catalog: CatalogServer[];
}

export interface ToolDiscoverResult {
  request: string;
  stage1_servers: { name: string; score: number }[];
  stage2_tools: Record<string, { server_score: number; tools: [string, number][] }>;
  final_tools: { name: string; server: string; score: number }[];
}
```

- [ ] **Step 2: Add API fetchers**

Append to `web/src/api.ts` (near `getEvalResults`), importing the new types at the top:

```typescript
export function getDebugTools(): Promise<DebugToolsResult> {
  return requestJson<DebugToolsResult>("/api/debug/tools");
}

export function discoverTools(query: string): Promise<ToolDiscoverResult> {
  return requestJson<ToolDiscoverResult>("/api/debug/tools/discover", {
    method: "POST",
    body: JSON.stringify({ query }),
  });
}
```

- [ ] **Step 3: Create the panel component**

Create `web/src/components/debug/ToolsPanel.tsx`:

```tsx
import { useEffect, useState } from "react";
import { discoverTools, getDebugTools } from "../../api";
import type { CatalogServer, ToolDiscoverResult } from "../../types";

/**
 * Dev-console panel: read-only view of the tool registry — the discovery
 * catalog grouped by server, plus a query box to test semantic tool discovery.
 */
export function ToolsPanel() {
  const [catalog, setCatalog] = useState<CatalogServer[] | null>(null);
  const [registeredCount, setRegisteredCount] = useState(0);
  const [query, setQuery] = useState("");
  const [discovery, setDiscovery] = useState<ToolDiscoverResult | null>(null);

  useEffect(() => {
    let alive = true;
    getDebugTools().then(
      (r) => {
        if (!alive) return;
        setCatalog(r.catalog);
        setRegisteredCount(r.registered.length);
      },
      () => alive && setCatalog([]),
    );
    return () => {
      alive = false;
    };
  }, []);

  function runDiscover() {
    if (!query.trim()) return;
    discoverTools(query).then(setDiscovery, () => setDiscovery(null));
  }

  return (
    <section className="tools-panel" aria-label="Tool registry">
      <h2>Tools</h2>
      <p className="tools-panel__count">{registeredCount} registered</p>
      {catalog?.map((server) => (
        <article key={server.name} className="tools-panel__server">
          <header>{server.name}</header>
          <ul>
            {server.tools.map((t) => (
              <li key={t.name}>
                <strong>{t.name}</strong> <span className="tools-panel__source">{t.source}</span>
                <div>{t.description}</div>
              </li>
            ))}
          </ul>
        </article>
      ))}
      <div className="tools-panel__discover">
        <input
          aria-label="Discovery query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Test discovery for a query…"
        />
        <button onClick={runDiscover}>Discover</button>
      </div>
      {discovery && (
        <ol className="tools-panel__results">
          {discovery.final_tools.map((t) => (
            <li key={t.name}>
              {t.name} <span>({t.server}, {t.score.toFixed(3)})</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
```

- [ ] **Step 4: Wire into DevConsole**

In `web/src/components/debug/DevConsole.tsx`, add the import and render it after `<EvalResultsPanel />`:

```tsx
import { ToolsPanel } from "./ToolsPanel";
```

```tsx
      <EvalResultsPanel />
      <ToolsPanel />
```

- [ ] **Step 5: Type-check and build**

Run: `cd web && npm run typecheck`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add web/src/types.ts web/src/api.ts web/src/components/debug/ToolsPanel.tsx web/src/components/debug/DevConsole.tsx
git commit -m "feat(dev-console): Tools panel for registry catalog + discovery"
```

---

## Self-Review

- **Spec coverage:** `GET /api/debug/tools` (Task 1) ✓; `POST /api/debug/tools/discover` + blank→422 (Task 2) ✓; catalog grouped by server ✓ (Task 1 asdict of `catalog_from_registry`); frontend panel + api + types + DevConsole wiring (Task 3) ✓; empty-registry safety ✓ (handler returns empty lists, no guard needed); non-goals respected (no invocation, no registry changes) ✓.
- **Placeholder scan:** none — all steps carry concrete code/commands.
- **Type consistency:** `getDebugTools`/`discoverTools` return `DebugToolsResult`/`ToolDiscoverResult` defined in Task 3 Step 1; `final_tools` shape matches `get_routing_details` (`{name, server, score}`); `registered` uses existing `ToolView`; backend keys (`registered`, `catalog`, `stage1_servers`, `final_tools`) match across handler and frontend types.
