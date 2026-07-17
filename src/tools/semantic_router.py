"""Hierarchical semantic routing for tool discovery.

Two-stage matching of a natural-language tool request to relevant tools:
1. Server-level routing: rank candidate servers by TF-IDF similarity.
2. Tool-level routing: rank tools within selected servers.

Inspired by MCP-Zero. Reduces search complexity while keeping precision.
"""

from __future__ import annotations

from dataclasses import dataclass


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
