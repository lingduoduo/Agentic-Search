"""FunctionTool wrappers for search and RAG, used as ToolAgentLoop routing tools."""

from __future__ import annotations

import json
import logging

from .base import FunctionTool, ToolEffect
from .search import search_tool

logger = logging.getLogger(__name__)

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
        if not results and pages:
            errors = [p.error for p in pages if p.error]
            return json.dumps({"error": errors[0] if errors else "No results returned"})
        return json.dumps(results)

    return FunctionTool(
        fn=_execute,
        name="search_routing_tool",
        description="Retrieve relevant documents from the corpus given a search query.",
        parameters=_SEARCH_TOOL_PARAMS,
        effect=ToolEffect.READ_ONLY,
    )


def build_rag_routing_tool(
    *,
    llm,
    search_url: str,
    top_k: int,
    filters=None,
) -> FunctionTool:
    """FunctionTool that generates a RAG answer."""

    async def _execute(query: str) -> str:
        try:
            from src.context import answer_with_retrieval

            result = await answer_with_retrieval(
                query,
                llm=llm,
                search_url=search_url,
                top_k=top_k,
                filters=filters,
            )
            return json.dumps({"answer": result.answer, "citations": result.citations})
        except Exception as exc:
            logger.error("rag_routing_tool failed: %s", exc, exc_info=True)
            return json.dumps({"error": str(exc)})

    return FunctionTool(
        fn=_execute,
        name="rag_routing_tool",
        description="Answer a question using retrieval-augmented generation.",
        parameters=_RAG_TOOL_PARAMS,
        effect=ToolEffect.READ_ONLY,
    )
