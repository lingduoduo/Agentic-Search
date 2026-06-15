"""FunctionTool wrappers for search and RAG, used as ToolAgentLoop routing tools."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .base import FunctionTool
from .search import search_tool

if TYPE_CHECKING:
    pass

# Populated on first call to avoid circular imports at module load time.
# Exposed at module level so tests can monkeypatch it.
answer_with_retrieval: Any = None

_SEARCH_TOOL_PARAMS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query."},
    },
    "required": ["query"],
}

_RAG_TOOL_PARAMS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The question to answer using retrieval.",
        },
    },
    "required": ["query"],
}


def build_search_routing_tool(*, search_url: str, top_k: int) -> FunctionTool:
    """FunctionTool that retrieves documents from the corpus."""

    async def _execute(query: str) -> str:
        pages = await search_tool(
            query,
            provider="retrieval",
            search_url=search_url,
            page_size=top_k,
        )
        results = [
            {"title": p.title or "", "content": p.summary or "", "url": p.url}
            for p in pages
            if not p.error
        ]
        return json.dumps(results)

    return FunctionTool(
        fn=_execute,
        name="search_routing_tool",
        description="Retrieve relevant documents from the corpus given a search query.",
        parameters=_SEARCH_TOOL_PARAMS,
    )


def build_rag_routing_tool(
    *,
    llm: "Any | None",
    search_url: str,
    top_k: int,
    filters: "Any | None" = None,
) -> FunctionTool:
    """FunctionTool that generates a RAG answer."""

    async def _execute(query: str) -> str:
        import src.tools.routing_tools as _self

        fn = _self.answer_with_retrieval
        if fn is None:
            from src.context import answer_with_retrieval as _awr

            _self.answer_with_retrieval = _awr
            fn = _awr
        result = await fn(
            query,
            llm=llm,
            search_url=search_url,
            top_k=top_k,
            filters=filters,
        )
        return json.dumps({"answer": result.answer, "citations": result.citations})

    return FunctionTool(
        fn=_execute,
        name="rag_routing_tool",
        description="Answer a question using retrieval-augmented generation.",
        parameters=_RAG_TOOL_PARAMS,
    )
