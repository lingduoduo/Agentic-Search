# Hierarchical Semantic Tool Router — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt the sampled MCP-Zero-style routing code into a working, tested `discover_tools()` capability over a real repo tool catalog, in a new module `src/tools/semantic_router.py`.

**Architecture:** Two-stage TF-IDF matching — rank servers, then rank tools within selected servers, combine scores. Catalog comes from a declarative default (the repo's real web_search / knowledge_base / answer tools) plus a `catalog_from_registry()` builder over the live `ToolRegistry`. The sampled block is removed from `routing_tools.py`.

**Tech Stack:** Python, `numpy`, `scikit-learn` (TfidfVectorizer, cosine_similarity — already a dependency at `requirements.txt:30`), `pytest`, `ruff`.

## Global Constraints

- New module starts with `from __future__ import annotations`; all imports at module top.
- Three environment servers in the default catalog: `web_search`, `knowledge_base`, `answer`. Tool membership exactly as in the table in Task 1.
- No `tool_knowledge_base` import; no bare `import config`. Config is a `RoutingConfig` dataclass; catalog types are local dataclasses.
- No attribute injection onto `ServerDefinition` (no `server._tool_embeddings`); per-server index lives inside the router.
- TF-IDF only — no dense embeddings, no new dependency.
- No wiring into `ToolAgentLoop` or the MCP server.
- `routing_tools.py`'s `build_search_routing_tool` / `build_rag_routing_tool` stay byte-for-byte unchanged; only the sampled `SemanticRouter`/`StructuredRequestParser` block is removed.
- `RoutingConfig` defaults (verbatim): `top_k_servers=3, top_k_tools=3, similarity_threshold=0.0, server_weight=0.3, tool_weight=0.7`.
- `source` labels are accurate: `"mcp"` only for the six tools actually exposed over MCP (`search_web`, `open_urls`, `search_indexed_documents`, `retrieve_documents`, `expand_query`, `ask_agentic_search`); `browser_search` is `"retrieval-server"`; `rag_routing_tool` is `"function"`. The `source="mcp"` set must equal the `docs/mcp.md` tool table.
- Documentation consolidation is scoped to `docs/mcp.md` only (no `docs/request-routing.md` change).
- `ruff check` / `ruff format` clean.

---

### Task 1: Catalog dataclasses + default catalog + registry builder

**Files:**
- Create: `src/tools/semantic_router.py`
- Test: `tests/unit/test_semantic_router.py`

**Interfaces:**
- Consumes: nothing (registry builder takes a duck-typed `registry` with `.list()` returning entries exposing `.source`, `.provider_id`, `.tool.name`, `.tool.schema.description`).
- Produces:
  - `@dataclass(frozen=True) ToolDefinition(name: str, description: str, source: str = "", server: str = "")`
  - `@dataclass ServerDefinition(name: str, description: str, tools: list[ToolDefinition])`
  - `default_tool_catalog() -> list[ServerDefinition]`
  - `catalog_from_registry(registry) -> list[ServerDefinition]`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_semantic_router.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

from src.tools.semantic_router import (
    ServerDefinition,
    ToolDefinition,
    catalog_from_registry,
    default_tool_catalog,
)


def test_default_catalog_has_three_named_servers_with_expected_tools():
    catalog = default_tool_catalog()
    by_name = {s.name: s for s in catalog}
    assert set(by_name) == {"web_search", "knowledge_base", "answer"}
    assert {t.name for t in by_name["web_search"].tools} == {
        "search_web",
        "open_urls",
        "browser_search",
    }
    assert {t.name for t in by_name["knowledge_base"].tools} == {
        "search_indexed_documents",
        "retrieve_documents",
        "expand_query",
    }
    assert {t.name for t in by_name["answer"].tools} == {
        "ask_agentic_search",
        "rag_routing_tool",
    }
    # Every tool records its owning server.
    for server in catalog:
        for tool in server.tools:
            assert tool.server == server.name
            assert tool.description


def _entry(name, desc, source, provider_id=None):
    tool = SimpleNamespace(name=name, schema=SimpleNamespace(description=desc))
    return SimpleNamespace(tool=tool, source=source, provider_id=provider_id)


def _fake_registry(entries):
    return SimpleNamespace(list=lambda: entries)


def test_catalog_from_registry_groups_by_provider_and_source():
    reg = _fake_registry(
        [
            _entry("get_forecast", "weather forecast", "openapi", "weather"),
            _entry("get_alerts", "weather alerts", "openapi", "weather"),
            _entry("greet", "say hi", "function", None),
        ]
    )
    by_name = {s.name: s for s in catalog_from_registry(reg)}
    assert set(by_name) == {"weather", "local"}
    assert [t.name for t in by_name["weather"].tools] == ["get_forecast", "get_alerts"]
    assert [t.name for t in by_name["local"].tools] == ["greet"]
    assert by_name["weather"].tools[0].source == "openapi"


def test_catalog_from_registry_empty_is_empty():
    assert catalog_from_registry(_fake_registry([])) == []


def _documented_mcp_tools() -> set[str]:
    """Tool names from the 'Tools available to the LLM client' table in docs/mcp.md.

    Consolidation guard: the catalog's MCP-sourced tools must match the doc.
    """
    from pathlib import Path
    import re

    doc = (Path(__file__).resolve().parents[2] / "docs" / "mcp.md").read_text()
    start = doc.index("## Tools available to the LLM client")
    section = doc[start : doc.index("\n## ", start + 1)]
    names: set[str] = set()
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        first_col = line.split("|")[1].strip()
        m = re.fullmatch(r"`([a-z_]+)`", first_col)
        if m:
            names.add(m.group(1))
    return names


def test_catalog_mcp_tools_match_docs_mcp_table():
    catalog = default_tool_catalog()
    catalog_mcp = {
        t.name
        for server in catalog
        for t in server.tools
        if t.source == "mcp"
    }
    assert catalog_mcp == _documented_mcp_tools()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_semantic_router.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.tools.semantic_router'`

- [ ] **Step 3: Write minimal implementation**

Create `src/tools/semantic_router.py`:

```python
"""Hierarchical semantic routing for tool discovery.

Two-stage matching of a natural-language tool request to relevant tools:
1. Server-level routing: rank candidate servers by TF-IDF similarity.
2. Tool-level routing: rank tools within selected servers.

Inspired by MCP-Zero. Reduces search complexity while keeping precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    source: str = ""
    server: str = ""


@dataclass
class ServerDefinition:
    name: str
    description: str
    tools: list[ToolDefinition]


def default_tool_catalog() -> list[ServerDefinition]:
    """The repo's real tool surface, grouped into domain servers."""
    return [
        ServerDefinition(
            name="web_search",
            description=(
                "Search the public internet for news, documentation, and general "
                "facts, and fetch full page content from URLs."
            ),
            tools=[
                ToolDefinition(
                    "search_web",
                    "Search the public internet via Google, SerpAPI, or Serper.",
                    "mcp",
                    "web_search",
                ),
                ToolDefinition(
                    "open_urls",
                    "Fetch the full text content of specific web page URLs.",
                    "mcp",
                    "web_search",
                ),
                ToolDefinition(
                    "browser_search",
                    "Browser-driven web search via playwright-cli when web APIs are unavailable.",
                    "retrieval-server",
                    "web_search",
                ),
            ],
        ),
        ServerDefinition(
            name="knowledge_base",
            description=(
                "Search and retrieve documents from the private indexed corpus."
            ),
            tools=[
                ToolDefinition(
                    "search_indexed_documents",
                    "Search the private knowledge base with optional document-set narrowing.",
                    "mcp",
                    "knowledge_base",
                ),
                ToolDefinition(
                    "retrieve_documents",
                    "Retrieve raw indexed document content and relevance scores.",
                    "mcp",
                    "knowledge_base",
                ),
                ToolDefinition(
                    "expand_query",
                    "Expand a query into BM25-optimised keyword variants for better recall.",
                    "mcp",
                    "knowledge_base",
                ),
            ],
        ),
        ServerDefinition(
            name="answer",
            description=(
                "Synthesize a grounded answer from retrieved evidence with citations."
            ),
            tools=[
                ToolDefinition(
                    "ask_agentic_search",
                    "Synthesize a cited answer from authenticated retrieved evidence.",
                    "mcp",
                    "answer",
                ),
                ToolDefinition(
                    "rag_routing_tool",
                    "Answer a question using retrieval-augmented generation.",
                    "function",
                    "answer",
                ),
            ],
        ),
    ]


def catalog_from_registry(registry) -> list[ServerDefinition]:
    """Group live ToolRegistry entries into servers.

    OpenAPI tools group by ``provider_id``; function tools go to a ``local``
    server. Preserves registry order within each server. Servers with no tools
    are omitted.
    """
    servers: dict[str, ServerDefinition] = {}
    for entry in registry.list():
        if entry.source == "openapi" and entry.provider_id:
            server_name = entry.provider_id
        else:
            server_name = "local"
        server = servers.get(server_name)
        if server is None:
            server = ServerDefinition(name=server_name, description=server_name, tools=[])
            servers[server_name] = server
        server.tools.append(
            ToolDefinition(
                name=entry.tool.name,
                description=entry.tool.schema.description,
                source=entry.source,
                server=server_name,
            )
        )
    return [s for s in servers.values() if s.tools]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_semantic_router.py -v`
Expected: PASS (6 tests, including the docs drift guard)

- [ ] **Step 5: Commit**

```bash
git add src/tools/semantic_router.py tests/unit/test_semantic_router.py
git commit -m "feat: add tool-discovery catalog dataclasses and builders"
```

---

### Task 2: RoutingConfig + SemanticRouter

**Files:**
- Modify: `src/tools/semantic_router.py`
- Test: `tests/unit/test_semantic_router.py`

**Interfaces:**
- Consumes: `ServerDefinition`, `ToolDefinition` (Task 1).
- Produces:
  - `@dataclass(frozen=True) RoutingConfig(top_k_servers=3, top_k_tools=3, similarity_threshold=0.0, server_weight=0.3, tool_weight=0.7)`
  - `class SemanticRouter(servers: list[ServerDefinition], config: RoutingConfig | None = None)`
  - `SemanticRouter.route_request(request: str, *, server_hint: str | None = None, top_k_servers: int | None = None, top_k_tools: int | None = None) -> list[ToolDefinition]`
  - `SemanticRouter.get_routing_details(request: str, *, server_hint=None, top_k_servers=None, top_k_tools=None) -> dict`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_semantic_router.py`:

```python
from src.tools.semantic_router import RoutingConfig, SemanticRouter


def test_router_ranks_web_tools_for_internet_request():
    router = SemanticRouter(default_tool_catalog())
    tools = router.route_request("search the public internet for recent news")
    assert tools, "expected at least one routed tool"
    assert tools[0].server == "web_search"


def test_router_ranks_knowledge_base_for_internal_docs_request():
    router = SemanticRouter(default_tool_catalog())
    tools = router.route_request("find internal indexed documents about FAISS")
    assert tools[0].server == "knowledge_base"


def _divergence_catalog():
    return [
        ServerDefinition(
            "weather",
            "weather forecast temperature climate",
            [ToolDefinition("lookup_weather", "get the current temperature", "", "weather")],
        ),
        ServerDefinition(
            "finance",
            "stock market finance trading",
            [ToolDefinition("lookup_stock", "get the current temperature", "", "finance")],
        ),
    ]


def test_server_hint_changes_stage1_winner():
    router = SemanticRouter(_divergence_catalog())
    # Without a hint the request text drives stage 1: "temperature" matches weather.
    no_hint = router.route_request("get the current temperature")
    assert no_hint[0].name == "lookup_weather"
    # With a hint the server-stage text picks finance instead.
    hinted = router.route_request(
        "get the current temperature", server_hint="stock market finance"
    )
    assert hinted[0].name == "lookup_stock"


def test_empty_catalog_routes_to_nothing():
    assert SemanticRouter([]).route_request("anything") == []


def test_threshold_filters_zero_similarity_requests():
    router = SemanticRouter(
        default_tool_catalog(), RoutingConfig(similarity_threshold=0.5)
    )
    # No shared vocabulary with any server/tool description.
    assert router.route_request("qwerty zxcvbn asdfgh") == []


def test_top_k_larger_than_catalog_is_clamped_and_deduped():
    router = SemanticRouter(
        default_tool_catalog(), RoutingConfig(top_k_servers=10, top_k_tools=10)
    )
    tools = router.route_request("search retrieve documents and answer questions")
    names = [t.name for t in tools]
    assert len(names) == len(set(names))  # no duplicates
    assert len(names) <= 8  # total tools in the default catalog


def test_routing_details_shape():
    router = SemanticRouter(default_tool_catalog())
    details = router.get_routing_details("search the public internet for news")
    assert details["request"] == "search the public internet for news"
    assert isinstance(details["stage1_servers"], list)
    assert details["stage1_servers"][0]["name"] == "web_search"
    assert "web_search" in details["stage2_tools"]
    assert details["final_tools"][0]["server"] == "web_search"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_semantic_router.py -k "router or hint or threshold or top_k or details or empty_catalog" -v`
Expected: FAIL — `ImportError: cannot import name 'RoutingConfig'` / `'SemanticRouter'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/tools/semantic_router.py` (after the catalog builders):

```python
@dataclass(frozen=True)
class RoutingConfig:
    top_k_servers: int = 3
    top_k_tools: int = 3
    similarity_threshold: float = 0.0
    server_weight: float = 0.3
    tool_weight: float = 0.7


class SemanticRouter:
    """Hierarchical semantic routing for tool discovery."""

    def __init__(
        self,
        servers: list[ServerDefinition],
        config: RoutingConfig | None = None,
    ):
        self.servers = servers
        self.config = config or RoutingConfig()
        self._server_vectorizer = None
        self._server_embeddings = None
        # server name -> (vectorizer, tool_embeddings)
        self._tool_index: dict[str, tuple] = {}
        self._build_server_index()
        self._build_tool_indices()

    @staticmethod
    def _fit(texts: list[str]):
        """Fit a TF-IDF index over *texts*, returning (vectorizer, matrix).

        Falls back to no stop-word filtering if that yields an empty vocabulary,
        and to (None, None) if the texts have no usable terms at all.
        """
        for stop_words in ("english", None):
            vectorizer = TfidfVectorizer(stop_words=stop_words)
            try:
                matrix = vectorizer.fit_transform(texts)
            except ValueError:
                continue
            return vectorizer, matrix
        return None, None

    @staticmethod
    def _similarity(vectorizer, matrix, query: str) -> np.ndarray:
        if vectorizer is None or matrix is None:
            return np.zeros(0)
        try:
            query_vec = vectorizer.transform([query])
        except ValueError:
            return np.zeros(matrix.shape[0])
        return cosine_similarity(query_vec, matrix)[0]

    def _build_server_index(self) -> None:
        if not self.servers:
            return
        descriptions = [f"{s.name} {s.description}" for s in self.servers]
        self._server_vectorizer, self._server_embeddings = self._fit(descriptions)

    def _build_tool_indices(self) -> None:
        for server in self.servers:
            if not server.tools:
                continue
            descriptions = [f"{t.name} {t.description}" for t in server.tools]
            vectorizer, embeddings = self._fit(descriptions)
            self._tool_index[server.name] = (vectorizer, embeddings)

    def _route_to_servers(self, query: str, top_k: int):
        sims = self._similarity(self._server_vectorizer, self._server_embeddings, query)
        if sims.size == 0:
            return []
        order = np.argsort(sims)[::-1][:top_k]
        return [(self.servers[i], float(sims[i])) for i in order]

    def _route_to_tools(self, server: ServerDefinition, query: str, top_k: int):
        index = self._tool_index.get(server.name)
        if index is None:
            return []
        vectorizer, embeddings = index
        sims = self._similarity(vectorizer, embeddings, query)
        if sims.size == 0:
            return []
        order = np.argsort(sims)[::-1][:top_k]
        return [(server.tools[i], float(sims[i])) for i in order]

    def _combined(self, server_score: float, tool_score: float) -> float:
        return (
            self.config.server_weight * server_score
            + self.config.tool_weight * tool_score
        )

    def _rank(self, request: str, server_hint: str | None, top_k_servers: int, top_k_tools: int):
        """Return [(tool, combined_score, server_name), ...] sorted deterministically."""
        server_query = server_hint if server_hint else request
        ranked_servers = self._route_to_servers(server_query, top_k_servers)
        scored = []
        for server, server_score in ranked_servers:
            for tool, tool_score in self._route_to_tools(server, request, top_k_tools):
                scored.append(
                    (tool, self._combined(server_score, tool_score), server.name)
                )
        # Score desc, then (server, tool name) for stable, deterministic ties.
        scored.sort(key=lambda x: (-x[1], x[2], x[0].name))
        return ranked_servers, scored

    def route_request(
        self,
        request: str,
        *,
        server_hint: str | None = None,
        top_k_servers: int | None = None,
        top_k_tools: int | None = None,
    ) -> list[ToolDefinition]:
        if not self.servers or self._server_embeddings is None:
            return []
        top_k_servers = self.config.top_k_servers if top_k_servers is None else top_k_servers
        top_k_tools = self.config.top_k_tools if top_k_tools is None else top_k_tools

        _servers, scored = self._rank(request, server_hint, top_k_servers, top_k_tools)

        seen: set[str] = set()
        result: list[ToolDefinition] = []
        for tool, score, _server_name in scored:
            if score < self.config.similarity_threshold or tool.name in seen:
                continue
            seen.add(tool.name)
            result.append(tool)
        return result[: top_k_tools * top_k_servers]

    def get_routing_details(
        self,
        request: str,
        *,
        server_hint: str | None = None,
        top_k_servers: int | None = None,
        top_k_tools: int | None = None,
    ) -> dict:
        top_k_servers = self.config.top_k_servers if top_k_servers is None else top_k_servers
        top_k_tools = self.config.top_k_tools if top_k_tools is None else top_k_tools

        ranked_servers, scored = self._rank(
            request, server_hint, top_k_servers, top_k_tools
        )

        stage2: dict[str, dict] = {}
        for server, server_score in ranked_servers:
            tools_scored = self._route_to_tools(server, request, top_k_tools)
            stage2[server.name] = {
                "server_score": server_score,
                "tools": [(t.name, s) for t, s in tools_scored],
            }

        seen: set[str] = set()
        final = []
        for tool, score, server_name in scored:
            if score < self.config.similarity_threshold or tool.name in seen:
                continue
            seen.add(tool.name)
            final.append({"name": tool.name, "server": server_name, "score": score})
        final = final[: top_k_tools * top_k_servers]

        return {
            "request": request,
            "stage1_servers": [
                {"name": s.name, "score": score} for s, score in ranked_servers
            ],
            "stage2_tools": stage2,
            "final_tools": final,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_semantic_router.py -v`
Expected: PASS (all Task 1 + Task 2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tools/semantic_router.py tests/unit/test_semantic_router.py
git commit -m "feat: add SemanticRouter with two-stage TF-IDF tool routing"
```

---

### Task 3: StructuredRequestParser + discover_tools

**Files:**
- Modify: `src/tools/semantic_router.py`
- Test: `tests/unit/test_semantic_router.py`

**Interfaces:**
- Consumes: `SemanticRouter`, `RoutingConfig`, `default_tool_catalog` (Tasks 1–2).
- Produces:
  - `class StructuredRequestParser` with `parse_request(text: str) -> dict | None` and `format_request(server_desc: str, tool_desc: str) -> str`
  - `discover_tools(request: str, *, catalog: list[ServerDefinition] | None = None, config: RoutingConfig | None = None) -> list[ToolDefinition]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_semantic_router.py`:

```python
from src.tools.semantic_router import StructuredRequestParser, discover_tools


def test_parser_round_trips():
    text = StructuredRequestParser.format_request("web platform", "fetch a page")
    parsed = StructuredRequestParser.parse_request(text)
    assert parsed == {"server": "web platform", "tool": "fetch a page"}


def test_parser_without_tags_returns_none():
    assert StructuredRequestParser.parse_request("just some text") is None


def test_parser_missing_tool_line_returns_none():
    text = "<tool_request>\nserver: web\n</tool_request>"
    assert StructuredRequestParser.parse_request(text) is None


def test_discover_tools_unstructured_web_request():
    tools = discover_tools("search the public internet for recent news")
    assert tools[0].server == "web_search"


def test_discover_tools_uses_structured_server_hint():
    request = StructuredRequestParser.format_request(
        "public web internet search", "fetch the full text of a web page url"
    )
    tools = discover_tools(request)
    assert tools[0].server == "web_search"


def test_discover_tools_empty_catalog():
    assert discover_tools("anything", catalog=[]) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_semantic_router.py -k "parser or discover" -v`
Expected: FAIL — `ImportError: cannot import name 'StructuredRequestParser'` / `'discover_tools'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/tools/semantic_router.py`:

```python
class StructuredRequestParser:
    """Parse and format MCP-Zero structured tool requests.

    Format::

        <tool_request>
        server: [platform/domain description]
        tool: [operation description]
        </tool_request>
    """

    OPEN = "<tool_request>"
    CLOSE = "</tool_request>"

    @staticmethod
    def parse_request(text: str) -> dict | None:
        if StructuredRequestParser.OPEN not in text or StructuredRequestParser.CLOSE not in text:
            return None
        start = text.find(StructuredRequestParser.OPEN) + len(StructuredRequestParser.OPEN)
        end = text.find(StructuredRequestParser.CLOSE)
        body = text[start:end].strip()

        result: dict[str, str] = {}
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("server:"):
                result["server"] = line[len("server:"):].strip()
            elif line.startswith("tool:"):
                result["tool"] = line[len("tool:"):].strip()

        if "server" in result and "tool" in result:
            return result
        return None

    @staticmethod
    def format_request(server_desc: str, tool_desc: str) -> str:
        return (
            f"{StructuredRequestParser.OPEN}\n"
            f"server: {server_desc}\n"
            f"tool: {tool_desc}\n"
            f"{StructuredRequestParser.CLOSE}"
        )


def discover_tools(
    request: str,
    *,
    catalog: list[ServerDefinition] | None = None,
    config: RoutingConfig | None = None,
) -> list[ToolDefinition]:
    """Discover the most relevant tools for a natural-language request.

    If *request* contains a ``<tool_request>`` block, the ``server:`` description
    drives server-stage routing and the ``tool:`` description drives tool-stage
    routing; otherwise the raw *request* drives both stages.
    """
    catalog = catalog if catalog is not None else default_tool_catalog()
    parsed = StructuredRequestParser.parse_request(request)
    if parsed is not None:
        server_hint: str | None = parsed["server"]
        tool_request = parsed["tool"]
    else:
        server_hint = None
        tool_request = request

    router = SemanticRouter(catalog, config)
    return router.route_request(tool_request, server_hint=server_hint)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_semantic_router.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/tools/semantic_router.py tests/unit/test_semantic_router.py
git commit -m "feat: add structured request parser and discover_tools entry point"
```

---

### Task 4: Remove the sampled block from routing_tools.py

**Files:**
- Modify: `src/tools/routing_tools.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `routing_tools.py` containing only its two FunctionTool builders (`build_search_routing_tool`, `build_rag_routing_tool`).

- [ ] **Step 1: Delete the sampled block**

In `src/tools/routing_tools.py`, delete everything from the standalone module
docstring that begins:

```python
"""
Hierarchical Semantic Routing for Tool Discovery.
```

through the end of the file (the `StructuredRequestParser.format_request` method
and its trailing lines). The file must end right after `build_rag_routing_tool`'s
`return FunctionTool(...)` block (the last line of the original
`build_rag_routing_tool`).

After deletion the file's imports (lines 1–11: `json`, `logging`, `FunctionTool`,
`ToolEffect`, `search_tool`) remain — all still used by the two builders. Do NOT
add or change anything else.

- [ ] **Step 2: Verify the module imports and is clean**

Run:
```bash
python3 -c "from src.tools.routing_tools import build_search_routing_tool, build_rag_routing_tool; print('ok')"
grep -c "SemanticRouter\|StructuredRequestParser\|tool_knowledge_base\|import config" src/tools/routing_tools.py
```
Expected: prints `ok`, then `0`.

- [ ] **Step 3: Confirm the router module still passes and lint is clean**

Run:
```bash
pytest tests/unit/test_semantic_router.py -q
ruff check src/tools/semantic_router.py src/tools/routing_tools.py tests/unit/test_semantic_router.py
ruff format --check src/tools/semantic_router.py src/tools/routing_tools.py tests/unit/test_semantic_router.py
```
Expected: all tests pass; ruff reports no errors.

- [ ] **Step 4: Commit**

```bash
git add src/tools/routing_tools.py
git commit -m "refactor: move sampled semantic router out of routing_tools.py"
```

---

### Task 5: Consolidate semantic tool discovery into docs/mcp.md

**Files:**
- Modify: `docs/mcp.md`

**Interfaces:**
- Consumes: `discover_tools`, `SemanticRouter`, `catalog_from_registry` (Tasks 1–3).
- Produces: documentation only.

- [ ] **Step 1: Add the "Semantic tool discovery" section**

In `docs/mcp.md`, insert the following new section immediately AFTER the
dynamic-mirroring paragraph (the line ending
"`... (src/internal/mcp_server/tools/dynamic.py).`") and BEFORE the `## Resources`
heading:

```markdown
## Semantic tool discovery (server-side)

As the exposed tool set grows, a caller or agent can narrow it to the most
relevant tools for a request instead of reasoning over the full list.
`src/tools/semantic_router.py` provides this as an optional, server-side helper:

- `discover_tools(request)` returns the tools most relevant to a natural-language
  request, using a two-stage TF-IDF match — first rank domain *servers*, then
  rank *tools* within the top servers, then combine the scores.
- The default catalog groups the real capabilities into three domain servers:

  | server | tools |
  |--------|-------|
  | `web_search` | `search_web`, `open_urls`, `browser_search` |
  | `knowledge_base` | `search_indexed_documents`, `retrieve_documents`, `expand_query` |
  | `answer` | `ask_agentic_search`, `rag_routing_tool` |

  `browser_search` is the standalone playwright-cli browser retrieval server, a
  routable capability that is **not** exposed as an MCP tool; the six tools in
  the [table above](#tools-available-to-the-llm-client) are the MCP surface.
- A structured request (`<tool_request>server: … tool: …</tool_request>`) routes
  the `server:` text through the server stage and the `tool:` text through the
  tool stage.

This does not change how MCP clients invoke tools — MCP tool selection stays
client-driven, as described above. Discovery is a ranking aid, not a dispatcher.
`catalog_from_registry()` builds the catalog from the live `ToolRegistry`, so any
tool mirrored to MCP via `sync_tool_to_mcp` (see the note above) also becomes
discoverable through the router.
```

- [ ] **Step 2: Verify the doc is consistent and links resolve**

Run:
```bash
grep -n "Semantic tool discovery" docs/mcp.md
python3 -c "import re,pathlib; d=pathlib.Path('docs/mcp.md').read_text(); assert d.index('Semantic tool discovery') < d.index('## Resources'); print('placement ok')"
pytest tests/unit/test_semantic_router.py::test_catalog_mcp_tools_match_docs_mcp_table -q
```
Expected: the grep finds the heading; "placement ok"; the drift-guard test still
passes (the new section does not alter the tools table).

- [ ] **Step 3: Commit**

```bash
git add docs/mcp.md
git commit -m "docs: consolidate semantic tool discovery into the MCP tools guide"
```

---

## Self-Review

**Spec coverage:**
- Catalog dataclasses + default catalog + `catalog_from_registry` → Task 1. ✓
- Accurate `source` labels (`browser_search`="retrieval-server") + docs drift-guard test → Task 1. ✓
- `RoutingConfig` + `SemanticRouter` (fixes: top imports, no `import config`, no attribute injection; robustness; two-field `server_hint`; `get_routing_details`) → Task 2. ✓
- `StructuredRequestParser` + `discover_tools` → Task 3. ✓
- Remove sampled block from `routing_tools.py`, builders untouched → Task 4. ✓
- Documentation consolidation in `docs/mcp.md` (Semantic tool discovery section, relates to tools table + dynamic-mirroring note) → Task 5. ✓
- Non-goals (no loop/MCP wiring, no dense, no new dep, no request-routing.md change) → enforced by Global Constraints. ✓
- Encoder seam (`_fit`/`_similarity`) → Task 2. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. The empty-vocabulary fallback in `_fit` is concrete.

**Type consistency:** `route_request`/`get_routing_details`/`_rank`/`_route_to_servers`/`_route_to_tools`/`_combined`/`_fit`/`_similarity` signatures and the `(tool, score, server_name)` tuple shape are consistent across Tasks 2–3. `discover_tools` uses `server_hint=` matching `route_request`. ✓
