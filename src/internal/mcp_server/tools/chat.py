"""Chat tool for the Agentic Search MCP server.

Exposes the full RAG pipeline as a single MCP tool so that any MCP client
(e.g. Claude Desktop) can ask a question and receive a grounded answer with
inline citations — without needing to chain search + open_urls manually.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.context import AnswerGenerationRequest
from src.context import CANONICAL_ABSTENTION
from src.context import VerificationStatus
from src.context import build_context_bundle
from src.context import generate_answer
from src.context.search import SearchResult
from src.internal.llm.interfaces import LLMConfig
from src.internal.llm.providers import OpenAICompatibleLLM

from ..api import mcp_server
from ..retrieval_client import authenticated_retrieve

logger = logging.getLogger(__name__)


def _result_metadata(result: Any | None = None) -> dict[str, Any]:
    if result is None:
        return {
            "confidence": 0.0,
            "verification_status": VerificationStatus.ABSTAINED.value,
            "abstained": True,
            "retry_count": 0,
            "tool_sources": [],
            "structured_output_requested": False,
            "structured_output_applied": False,
            "structured_output_downgraded": False,
            "structured_output_category": None,
        }
    status = getattr(result, "verification_status", None)
    tool_evidence = getattr(result, "tool_evidence", [])
    if not isinstance(tool_evidence, list):
        tool_evidence = []
    retry_count = getattr(result, "retry_count", 0)
    if not isinstance(retry_count, int):
        retry_count = 0
    requested = getattr(result, "structured_output_requested", False)
    applied = getattr(result, "structured_output_applied", False)
    downgraded = getattr(result, "structured_output_downgraded", False)
    category = getattr(result, "structured_output_category", None)
    return {
        "confidence": getattr(result, "confidence", None),
        "verification_status": getattr(status, "value", status),
        "abstained": getattr(result, "abstained", False),
        "retry_count": retry_count,
        "tool_sources": [
            {"name": item.tool_name} for item in tool_evidence if item.tool_name
        ],
        "structured_output_requested": requested
        if isinstance(requested, bool)
        else False,
        "structured_output_applied": applied if isinstance(applied, bool) else False,
        "structured_output_downgraded": downgraded
        if isinstance(downgraded, bool)
        else False,
        "structured_output_category": category if isinstance(category, str) else None,
    }


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
    from retrieved document text. This reduces synthesis risk but is not a guarantee
    that the source evidence itself is correct or complete.

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

    try:
        documents = await authenticated_retrieve(question, top_k=top_k)
        context = build_context_bundle(
            question,
            [
                SearchResult(
                    contents=document.content,
                    score=document.score,
                    title=document.title,
                    url=document.url,
                    metadata=document.metadata,
                )
                for document in documents
                if document.content.strip()
            ],
        )
        if not context.documents:
            return {
                "answer": CANONICAL_ABSTENTION,
                "citations": [],
                "sources": [],
                **_result_metadata(),
            }

        llm = _build_llm()
        if llm is None:
            logger.info("MCP Server: no LLM configured — using extractive fallback")
        result = generate_answer(
            AnswerGenerationRequest(
                question=question,
                context=context,
                verify_grounding=True,
            ),
            llm=llm,
        )
    except Exception as exc:
        logger.error("MCP Server: ask_agentic_search failed: %s", exc, exc_info=True)
        return {
            "error": "Unable to answer from available evidence.",
            "answer": CANONICAL_ABSTENTION,
            "citations": [],
            "sources": [],
            **_result_metadata(),
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
        **_result_metadata(result),
    }
