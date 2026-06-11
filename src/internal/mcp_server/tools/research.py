"""Research tools for the Agentic Search MCP server.

Provides higher-level tools that go beyond single-pass RAG:

  deep_research       — multi-round iterative RAG via AgenticRAGLoop; uses
                        gap-analysis prompts to generate follow-up queries until
                        evidence is sufficient or max_rounds is reached.

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

from src.agents.agentic_rag import AgenticRAGConfig, AgenticRAGLoop
from src.internal.llm.interfaces import LLMConfig
from src.internal.llm.providers import OpenAICompatibleLLM
from src.internal.servers.secondary_llm_flows.query_expansion import expand_keywords
from src.context.pipeline import retrieve_context

from ..api import mcp_server

logger = logging.getLogger(__name__)


def _retrieval_url() -> str:
    port = os.getenv("AGENTIC_SEARCH_RETRIEVAL_PORT", "8000")
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
async def deep_research(
    question: str,
    top_k: int = 5,
    max_rounds: int = 3,
) -> dict[str, Any]:
    """
    Answer a question using multi-round iterative RAG with gap analysis.

    Unlike ``ask_agentic_search`` (single retrieval pass), this tool iterates:
      1. Retrieve documents for the question.
      2. Ask the LLM: is the evidence sufficient?  If yes → synthesize and return.
      3. If not, perform gap analysis to identify missing information and generate
         follow-up queries, then repeat up to *max_rounds* times.

    Use this tool for complex, multi-faceted questions where a single retrieval
    pass may miss relevant details.  Typical wall-time is 2–10 seconds depending
    on *max_rounds* and retrieval latency.

    Returns::
        {
            "answer":      str,        # grounded answer with [D1]… citations
            "citations":   [str],      # citation labels extracted from the answer
            "rounds_used": int,        # how many retrieval rounds were needed
            "sources": [               # all documents accumulated across rounds
                {"title": str, "url": str, "content": str},
                ...
            ]
        }

    Example usage::
        {"question": "How does FAISS compare to Annoy for billion-scale search?", "max_rounds": 3}
    """
    logger.info(
        "MCP Server: deep_research: question=%r top_k=%d max_rounds=%d",
        question,
        top_k,
        max_rounds,
    )

    llm = _build_llm()
    loop = AgenticRAGLoop(
        AgenticRAGConfig(
            max_rounds=max_rounds,
            topk=top_k,
            retrieval_url=_retrieval_url(),
        ),
        llm=llm,
    )

    try:
        result = await loop.run(question)
    except Exception as exc:
        logger.error("MCP Server: deep_research failed: %s", exc, exc_info=True)
        return {
            "error": str(exc),
            "answer": "",
            "citations": [],
            "rounds_used": 0,
            "sources": [],
        }

    sources = [
        {"title": doc.title, "url": doc.url or "", "content": doc.content}
        for doc in result.context.documents
    ]
    return {
        "answer": result.answer,
        "citations": result.citations,
        "rounds_used": result.rounds_used,
        "sources": sources,
    }


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
