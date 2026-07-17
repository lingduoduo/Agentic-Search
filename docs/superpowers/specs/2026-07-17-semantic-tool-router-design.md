# Hierarchical Semantic Router for Tool Discovery — Design

**Date:** 2026-07-17
**Branch:** `feat/semantic-tool-router`
**Status:** Approved (brainstorming)

## Motivation

Sampled MCP-Zero-style routing code was added to `src/tools/routing_tools.py`
(`SemanticRouter` + `StructuredRequestParser`). As dropped in it does not run: it
imports a nonexistent `tool_knowledge_base` module (`ServerDefinition`,
`ToolDefinition`) and a nonexistent `config` module, injects state onto foreign
objects (`server._tool_embeddings`), and has imports in the middle of the file.

We adapt it into a working, tested tool-discovery capability grounded in this
repo: `discover_tools(request)` returns the most relevant tools via two-stage
server→tool TF-IDF matching over a real catalog. The catalog is grounded in the
repo's actual tool surface (the MCP web-search / knowledge-base / answer tools),
and can also ingest live `ToolRegistry` entries so it scales as tools register.

## Scope

### In scope
- New module `src/tools/semantic_router.py`: catalog dataclasses, default
  catalog, registry catalog builder, `SemanticRouter`, `StructuredRequestParser`,
  and the `discover_tools` entry point.
- Remove the sampled `SemanticRouter` / `StructuredRequestParser` block from
  `src/tools/routing_tools.py` (leaving its `build_search_routing_tool` /
  `build_rag_routing_tool` untouched).
- New `tests/unit/test_semantic_router.py`, including a **drift-guard test** that
  ties the catalog's MCP-sourced tools to the documented tool table in
  `docs/mcp.md`.
- **Documentation consolidation in `docs/mcp.md`:** a new "Semantic tool
  discovery" section describing the server-side router and relating it to the
  existing "MCP tool selection is independent" note and the dynamic-tool
  mirroring note. This is the doc where tool setup and tool selection already
  live; the new mechanism belongs beside them.

### Non-goals
- No wiring into `ToolAgentLoop` or the MCP server (the "full integration"
  option was explicitly not chosen).
- No dense embeddings — TF-IDF only (sklearn is already a dependency).
- No new dependency.
- No `rewrite` of the FunctionTool builders in `routing_tools.py`.
- No changes to `docs/request-routing.md` (consolidation is scoped to
  `docs/mcp.md`).

### File placement
The router is a distinct responsibility from `routing_tools.py`'s FunctionTool
builders, so it lives in its own module `src/tools/semantic_router.py`. The
sampled block is removed from `routing_tools.py`.

## Components (`src/tools/semantic_router.py`)

### 1. Catalog dataclasses (replace the phantom `tool_knowledge_base`)

```python
@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    source: str = ""     # e.g. "mcp", "function", "openapi"
    server: str = ""     # owning server name (filled by the catalog builder)

@dataclass
class ServerDefinition:
    name: str
    description: str
    tools: list[ToolDefinition]
```

`ServerDefinition` is mutable but the router does NOT inject embedding state onto
it (unlike the sample). Embeddings live inside the router.

### 2. Default catalog — `default_tool_catalog() -> list[ServerDefinition]`

Declarative, grounded in the repo's real tools:

| server (description) | tools |
|---|---|
| `web_search` — public internet search and page fetching | `search_web`, `open_urls`, `browser_search` |
| `knowledge_base` — private indexed corpus retrieval | `search_indexed_documents`, `retrieve_documents`, `expand_query` |
| `answer` — grounded answer synthesis over retrieved evidence | `ask_agentic_search`, `rag_routing_tool` |

Tool descriptions reuse the real one-line summaries from the MCP tool
docstrings / `docs/mcp.md`. Each `ToolDefinition.server` is set to its owning
server name. **`source` labels are accurate to what the investigation found**:
- `source="mcp"` — the six tools actually exposed over MCP (`search_web`,
  `open_urls`, `search_indexed_documents`, `retrieve_documents`, `expand_query`,
  `ask_agentic_search`). These must exactly match the `docs/mcp.md` tool table.
- `source="retrieval-server"` — `browser_search`, the standalone playwright-cli
  browser retrieval server. It is a real routable capability but **not** an
  MCP-exposed tool, so it is labelled honestly and excluded from the drift guard.
- `source="function"` — `rag_routing_tool`, a `FunctionTool` builder in
  `routing_tools.py`.

### 3. `catalog_from_registry(registry) -> list[ServerDefinition]`

Ingests `registry.list()` (`list[ToolEntry]`, each `ToolEntry(tool, source,
provider_id)`). Groups entries into servers:
- OpenAPI tools (`source == "openapi"`): grouped by `provider_id` → one server
  per provider (server name = `provider_id`).
- Function tools (`source == "function"`): grouped into a single `"local"`
  server.

Each `entry.tool` maps to `ToolDefinition(name=tool.name,
description=tool.schema.description, source=entry.source, server=<server name>)`.
Servers with no tools are omitted.

### 4. `RoutingConfig` (replace `import config`)

```python
@dataclass(frozen=True)
class RoutingConfig:
    top_k_servers: int = 3
    top_k_tools: int = 3
    similarity_threshold: float = 0.0   # 0.0 keeps all top-k; raise to prune
    server_weight: float = 0.3
    tool_weight: float = 0.7
```

### 5. `SemanticRouter`

Cleaned + enhanced version of the sampled class.

**Construction:** `SemanticRouter(servers: list[ServerDefinition], config:
RoutingConfig | None = None)`. Precomputes a server-level TF-IDF index and, per
server, a `(vectorizer, tool_embeddings)` pair stored in an internal dict keyed
by server name — no attribute injection onto `ServerDefinition`.

**Fixes vs the sample:**
- All imports at module top (`numpy`, sklearn).
- No `import config`; uses `RoutingConfig`.
- No `server._tool_embeddings`; per-server index held in `self._tool_index:
  dict[str, tuple[TfidfVectorizer, sparse_matrix]]`.
- Combine-score weights come from `RoutingConfig` (0.3 / 0.7 defaults).

**Robustness:**
- Empty catalog, or every server empty → routing returns `[]`.
- A server with no tools is skipped.
- An all-zero-similarity request (no shared TF-IDF terms) yields scores of 0.0;
  results are then filtered by `similarity_threshold` (default 0.0 keeps them).
- `top_k` larger than available is clamped by slicing.
- Deterministic tie-break: stable sort; when combined scores tie, order by
  `(server_name, tool_name)`.
- Dedup by tool name across servers (keep the highest-scoring occurrence).

**Methods:**
- `route_request(request: str, *, server_hint: str | None = None,
  top_k_servers=None, top_k_tools=None) -> list[ToolDefinition]`
  - **Two-field enhancement:** `server_hint` (the `server:` description from a
    structured request) drives stage-1 server ranking; `request` (the `tool:`
    description) drives stage-2 tool ranking. When `server_hint` is `None`, the
    single `request` string drives both stages (sample behavior).
  - Stage 1: rank servers by cosine similarity to the server-stage text; take
    top `top_k_servers`.
  - Stage 2: within each selected server, rank tools by cosine similarity to the
    tool-stage text; take top `top_k_tools`.
  - Combine `combined = server_weight * server_score + tool_weight * tool_score`,
    sort desc, filter by `similarity_threshold`, dedup by tool name, return the
    top `top_k_tools * top_k_servers` `ToolDefinition`s.
- `get_routing_details(request, *, server_hint=None, ...) -> dict` — the
  observability view: `{request, stage1_servers:[{name,score}],
  stage2_tools:{server:{server_score, tools:[(name,score)]}}, final_tools:
  [{name, server, score}]}`.

### 6. `StructuredRequestParser`

Kept, parsing the MCP-Zero structured request:

```
<tool_request>
server: [platform/domain description]
tool: [operation description]
</tool_request>
```

- `parse_request(text) -> dict | None` — returns `{"server":…, "tool":…}` when
  both fields are present inside the tags, else `None`.
- `format_request(server_desc, tool_desc) -> str` — the inverse.

### 7. `discover_tools(request, *, catalog=None, config=None) -> list[ToolDefinition]`

Primary public API:
1. `catalog = catalog or default_tool_catalog()`.
2. If `request` contains a `<tool_request>` block, parse it: use `server` as
   `server_hint` and `tool` as the stage-2 request; otherwise use the raw
   `request` for both stages.
3. Build a `SemanticRouter(catalog, config)` and return `route_request(...)`.

### Encoder seam
TF-IDF fit/transform + cosine similarity are confined to two small private
helpers (`_fit(texts)` / `_similarity(vectorizer, matrix, query)`) so a dense
encoder could replace them later. No public interface is introduced now (YAGNI).

### 8. Documentation consolidation (`docs/mcp.md`)

`docs/mcp.md` already documents tool setup (the "Tools available to the LLM
client" table) and tool selection (the note that MCP tool selection is client-
driven, and the dynamic-mirroring note). The new server-side router belongs
beside them. Add a **"Semantic tool discovery"** subsection after the tools
table that:
- Describes `discover_tools(request)` / `SemanticRouter`: a server-side,
  two-stage (server → tool) TF-IDF matcher that narrows a large tool set to the
  most relevant tools for a natural-language request. It is an **optional helper
  for callers/agents**, not a change to how MCP clients invoke tools — it does
  not contradict the existing "MCP tool selection is independent of the web UI's
  auto-router" statement, which stays true.
- Shows the domain-server catalog (`web_search` / `knowledge_base` / `answer`)
  and notes `browser_search` is the provider-backed browser retrieval server,
  not an MCP tool.
- Ties `catalog_from_registry` to the existing dynamic-mirroring note
  (`sync_tool_to_mcp`): the same live `ToolRegistry` that feeds MCP also feeds
  the router's catalog, so dynamically registered tools become discoverable.

No behavioral doc claims change; the tools table stays the source of truth for
the MCP surface, and the drift-guard test keeps the catalog aligned with it.

## Tests (`tests/unit/test_semantic_router.py`)

- `default_tool_catalog()` returns the three named servers with the listed tools.
- `catalog_from_registry`: register two fake tools (one `openapi` with
  `provider_id="weather"`, one `function`) → assert a `weather` server and a
  `local` server with the right tools.
- Routing relevance: "search the public internet for recent news" ranks
  `search_web`/`open_urls` above `knowledge_base` tools; "find internal indexed
  documents about FAISS" ranks `search_indexed_documents`/`retrieve_documents`
  on top.
- Two-field routing: a structured request whose `server:` desc points at the
  knowledge base but whose `tool:` desc says "fetch a web page" — assert the
  `server_hint` changes stage-1 selection vs the single-string path.
- Parser: `format_request` → `parse_request` round-trips; text without the tags
  returns `None`; text missing the `tool:` line returns `None`.
- Edges: empty catalog → `[]`; a request with no shared terms and
  `similarity_threshold=0.5` → `[]`; `top_k` larger than the catalog is clamped.
- **Drift guard:** parse the "Tools available to the LLM client" table in
  `docs/mcp.md`, extract the documented MCP tool names, and assert they exactly
  equal the set of `source="mcp"` tool names in `default_tool_catalog()`. Fails
  if the catalog or the doc drifts. (`browser_search`/`rag_routing_tool` are
  excluded since they are not `source="mcp"`.)

## Success criteria

1. `from src.tools.semantic_router import discover_tools` works; `discover_tools`
   returns sensible `ToolDefinition`s for web vs knowledge-base requests.
2. `pytest tests/unit/test_semantic_router.py` passes.
3. `routing_tools.py` no longer contains `SemanticRouter`/`StructuredRequestParser`
   and still imports/works (its FunctionTool builders unchanged).
4. No `tool_knowledge_base` or bare `import config` anywhere; `ruff` clean.
5. `docs/mcp.md` has a "Semantic tool discovery" section consolidating the new
   mechanism with the existing tools table and dynamic-mirroring note; the
   drift-guard test passes (catalog MCP tools == documented MCP tools).
