"""Chat tool for the Agentic Search MCP server.

Exposes the full RAG pipeline as a single MCP tool so that any MCP client
(e.g. Claude Desktop) can ask a question and receive a grounded answer with
inline citations — without needing to chain search + open_urls manually.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.context import answer_with_retrieval
from src.internal.llm.interfaces import LLMConfig
from src.internal.llm.providers import OpenAICompatibleLLM

from ..api import mcp_server

logger = logging.getLogger(__name__)


def _retrieval_url() -> str:
    port = os.getenv("AGENTIC_SEARCH_RETRIEVAL_PORT", "8000")
    return f"http://localhost:{port}/retrieve"


def _build_llm() -> OpenAICompatibleLLM | None:
    """Return an LLM client from env vars, or None to use extractive fallback."""
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
async def ask_agentic_search(
    question: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Ask Agentic Search a question using the full retrieval-augmented generation pipeline.

    Retrieves relevant documents from the indexed knowledge base, synthesizes a
    grounded answer, and returns inline citations. Use this tool when you want a
    direct answer rather than raw search results.

    If no LLM API key is configured, an extractive fallback answer is returned
    using only the retrieved document text — no hallucination risk.

    Returns::
        {
            "answer": str,          # grounded answer with [D1], [D2] … citation markers
            "citations": [str],     # citation labels extracted from the answer
            "sources": [            # retrieved documents used for grounding
                {"title": str, "url": str, "content": str},
                ...
            ]
        }

    Example usage::
        {"question": "What indexing methods does Agentic Search support?", "top_k": 5}
    """
    logger.info("MCP Server: ask_agentic_search: question=%r top_k=%d", question, top_k)

    llm = _build_llm()
    if llm is None:
        logger.info("MCP Server: no LLM configured — using extractive fallback")

    try:
        result = await answer_with_retrieval(
            question,
            llm=llm,
            search_url=_retrieval_url(),
            top_k=top_k,
        )
    except Exception as exc:
        logger.error("MCP Server: ask_agentic_search failed: %s", exc, exc_info=True)
        return {
            "error": str(exc),
            "answer": "",
            "citations": [],
            "sources": [],
        }

    sources = [
        {
            "title": doc.title,
            "url": doc.url or "",
            "content": doc.content,
        }
        for doc in result.context.documents
    ]

    return {
        "answer": result.answer,
        "citations": result.citations,
        "sources": sources,
    }
