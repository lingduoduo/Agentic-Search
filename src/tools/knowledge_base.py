"""Built-in executable tools that seed the global ToolRegistry.

The ToolRegistry is the single source of truth for this process's runnable
tools. ``tool_knowledge_base()`` is the built-in seed set; ``seed_tools()``
registers it. OpenAPI tools are added separately at runtime via
``register_from_openapi``. MCP-native tools live in the MCP server process and
are not registered here.
"""

from __future__ import annotations

import os

from .base import Tool
from .registry import ToolRegistry
from .routing_tools import build_rag_routing_tool, build_search_routing_tool
from .search import MultiQueryWebSearchTool, build_search_tool, make_web_cascade_search

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
            search_fn=make_web_cascade_search(
                browser_search_url=os.getenv("AGENTIC_SEARCH_BROWSER_SEARCH_URL")
            ),
            page_size=top_k,
        ),
        build_search_tool(provider="retrieval", search_url=search_url, page_size=top_k),
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
