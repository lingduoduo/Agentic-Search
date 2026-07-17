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
            server = ServerDefinition(
                name=server_name, description=server_name, tools=[]
            )
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


def get_all_tools(servers: list[ServerDefinition]) -> list[ToolDefinition]:
    """Flatten a catalog into a single list of every tool across all servers."""
    all_tools: list[ToolDefinition] = []
    for server in servers:
        all_tools.extend(server.tools)
    return all_tools


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
        order = np.argsort(sims, kind="stable")[::-1][:top_k]
        return [(self.servers[i], float(sims[i])) for i in order]

    def _route_to_tools(self, server: ServerDefinition, query: str, top_k: int):
        index = self._tool_index.get(server.name)
        if index is None:
            return []
        vectorizer, embeddings = index
        sims = self._similarity(vectorizer, embeddings, query)
        if sims.size == 0:
            return []
        order = np.argsort(sims, kind="stable")[::-1][:top_k]
        return [(server.tools[i], float(sims[i])) for i in order]

    def _combined(self, server_score: float, tool_score: float) -> float:
        return (
            self.config.server_weight * server_score
            + self.config.tool_weight * tool_score
        )

    def _rank(
        self,
        request: str,
        server_hint: str | None,
        top_k_servers: int,
        top_k_tools: int,
    ):
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
        top_k_servers = (
            self.config.top_k_servers if top_k_servers is None else top_k_servers
        )
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
        top_k_servers = (
            self.config.top_k_servers if top_k_servers is None else top_k_servers
        )
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
        if (
            StructuredRequestParser.OPEN not in text
            or StructuredRequestParser.CLOSE not in text
        ):
            return None
        start = text.find(StructuredRequestParser.OPEN) + len(
            StructuredRequestParser.OPEN
        )
        end = text.find(StructuredRequestParser.CLOSE)
        body = text[start:end].strip()

        result: dict[str, str] = {}
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("server:"):
                result["server"] = line[len("server:") :].strip()
            elif line.startswith("tool:"):
                result["tool"] = line[len("tool:") :].strip()

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
