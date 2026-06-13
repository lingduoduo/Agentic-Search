"""Research tools for the Agentic Search MCP server.

retrieve_documents  — raw document retrieval; returns title, url, content and
                      score for each result so the caller can inspect them
                      before generating an answer.

expand_query        — LLM-backed keyword expansion; produces BM25-optimised
                      alternative queries that improve recall when passed to
                      search_indexed_documents.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.internal.llm.interfaces import LLMConfig
from src.internal.llm.providers import OpenAICompatibleLLM
from src.internal.servers.secondary_llm_flows.query_expansion import expand_keywords
from src.context.pipeline import retrieve_context

from ..api import mcp_server

logger = logging.getLogger(__name__)


def _retrieval_url() -> str:
    port = os.getenv("AGENTIC_SEARCH_RETRIEVAL_PORT", "8001")
    return f"http://localhost:{port}/retrieve"


def _build_llm() -> OpenAICompatibleLLM | None:
    """Return an LLM client built from env vars, or None."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AGENTIC_SEARCH_LLM_API_KEY")
    if not api_key:
        return None
    return OpenAICompatibleLLM(
        LLMConfig(
            model_provider=os.getenv("AGENTIC_SEARCH_LLM_PROVIDER", "openai"),
            model_name=os.getenv("AGENTIC_SEARCH_LLM_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            api_base=os.getenv("AGENTIC_SEARCH_LLM_API_BASE"),
        )
    )


@mcp_server.tool()
async def retrieve_documents(
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Retrieve raw documents from the indexed knowledge base without generating an answer.

    Use this tool when you want to inspect the retrieved evidence yourself before
    composing an answer, or when you need the source documents for a downstream task
    (e.g. summarisation, comparison, extraction).

    Different from ``search_indexed_documents``: returns the full document content
    and relevance score rather than a trimmed snippet list.

    Returns::
        {
            "documents": [
                {
                    "id":      str,   # ephemeral result ID (D1, D2, …)
                    "title":   str,
                    "url":     str | null,
                    "content": str,   # full chunk content
                    "score":   float  # relevance score (higher is better)
                },
                ...
            ],
            "query": str
        }

    Example usage::
        {"query": "transformer attention mechanism", "top_k": 8}
    """
    logger.info("MCP Server: retrieve_documents: query=%r top_k=%d", query, top_k)

    try:
        bundle = await retrieve_context(
            query,
            search_url=_retrieval_url(),
            top_k=top_k,
        )
    except Exception as exc:
        logger.error("MCP Server: retrieve_documents failed: %s", exc, exc_info=True)
        return {"error": str(exc), "documents": [], "query": query}

    documents = [
        {
            "id": doc.id,
            "title": doc.title,
            "url": doc.url,
            "content": doc.content,
            "score": doc.score,
        }
        for doc in bundle.documents
    ]
    return {"documents": documents, "query": query}


@mcp_server.tool()
async def expand_query(
    query: str,
) -> dict[str, Any]:
    """
    Expand a search query into multiple BM25-optimised keyword variants.

    Useful before calling ``search_indexed_documents``: run this tool first,
    then search for each expanded query and merge the results for better recall.

    Requires an LLM API key (OPENAI_API_KEY or AGENTIC_SEARCH_LLM_API_KEY).
    Returns the original query unchanged if no LLM is configured.

    Returns::
        {
            "original":  str,
            "expanded":  [str, ...],   # LLM-generated keyword variants (may be empty)
            "all":       [str, ...]    # original + expanded, deduplicated
        }

    Example usage::
        {"query": "how does retrieval augmented generation work"}
    """
    logger.info("MCP Server: expand_query: query=%r", query)

    llm = _build_llm()
    if llm is None:
        logger.info("MCP Server: expand_query — no LLM, returning original query only")
        return {"original": query, "expanded": [], "all": [query]}

    try:
        expanded = expand_keywords(query, llm)
    except Exception as exc:
        logger.warning("MCP Server: expand_query failed: %s", exc)
        expanded = []

    seen: set[str] = {query.lower()}
    deduped = []
    for q in expanded:
        if q.lower() not in seen:
            seen.add(q.lower())
            deduped.append(q)

    return {
        "original": query,
        "expanded": deduped,
        "all": [query] + deduped,
    }
