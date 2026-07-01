"""Intent routing helpers for the /api/agent endpoint."""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.base import AgentLoopOutput
    from src.context.models import LLMClient

logger = logging.getLogger(__name__)

_SEARCH_RE = re.compile(
    r"\b(find|list|retrieve|search for|show me|pull|get me|look up|fetch)\b",
    re.IGNORECASE,
)
_CHAT_RE = re.compile(
    r"\b(explain|summarize|help me|write|what is|how do|why|difference between|compare|describe)\b",
    re.IGNORECASE,
)
_VERB_RE = re.compile(
    r"\b(is|are|was|were|do|does|did|have|has|can|could|would|should|will)\b",
    re.IGNORECASE,
)


def _infer_intent_from_output(output: "AgentLoopOutput") -> str:
    """Infer search/chat/tool intent from the first tool called in the output.

    Only the first line of action_trace is examined; subsequent tool calls
    do not affect intent classification.
    """
    if not output.action_trace:
        return "chat"
    first_line = output.action_trace.split("\n")[0].strip()
    try:
        record = json.loads(first_line)
        tool_name = record.get("tool_name", "")
        if tool_name == "search_routing_tool":
            return "search"
        if tool_name == "rag_routing_tool":
            return "chat"
        if tool_name:
            return "tool"
    except (json.JSONDecodeError, AttributeError):
        pass
    return "chat"


# ---------------------------------------------------------------------------
# 4-way agentic routing: pick the strategy, then the web layer dispatches the
# matching agent loop (direct LLM / agentic RAG / multi-turn search / tool use).
# ---------------------------------------------------------------------------


class RouteStrategy(str, Enum):
    """High-level agent strategy chosen by the entry-point router."""

    DIRECT_LLM = "direct_llm"  # parametric answer, no retrieval
    AGENTIC_RAG = "agentic_rag"  # query decompose + HyDE + grounded synthesis
    SEARCH_AGENT = "search_agent"  # multi-turn search until evidence suffices
    TOOL_AGENT = "tool_agent"  # OpenAPI / MCP function calling


# Imperative verbs that imply taking an action through a tool/MCP.
_TOOL_RE = re.compile(
    r"\b(send|email|create|open (?:a|an) (?:ticket|issue|pr)|file (?:a|an) "
    r"(?:ticket|issue)|schedule|book|call the api|invoke|run the|execute|"
    r"post to|update the|delete the|add to)\b",
    re.IGNORECASE,
)
# Conversational / generative asks that need no retrieval.
_DIRECT_RE = re.compile(
    r"\b(write|translate|rephrase|reword|rewrite|draft|brainstorm|"
    r"hello|hi there|thanks|joke|poem|haiku)\b",
    re.IGNORECASE,
)


def _is_bare_lookup(query: str) -> bool:
    """True for a short, verb-less term/entity, e.g. "FAISS", "vector database".

    Such a query is unambiguously a grounded lookup, so it routes to
    SEARCH_AGENT deterministically rather than risking the LLM classifier
    sending it to direct_llm (ungrounded). Anything carrying a tool, search,
    generative, conversational, question, or auxiliary-verb signal is excluded —
    those are handled by the normal cascade / classifier.
    """
    q = query.strip()
    if not q or q.endswith("?"):
        return False
    if (
        _TOOL_RE.search(q)
        or _SEARCH_RE.search(q)
        or _DIRECT_RE.search(q)
        or _CHAT_RE.search(q)
        or _VERB_RE.search(q)
    ):
        return False
    return len(q.split()) <= 3


def _rule_based_route(query: str) -> RouteStrategy:
    """Heuristic 4-way route. Precedence: tool > search > direct-llm > lookup > rag.

    The default is AGENTIC_RAG: when no signal dominates, a grounded answer is
    safer than an ungrounded one.
    """
    q = query.strip()
    if not q:
        return RouteStrategy.AGENTIC_RAG
    if _TOOL_RE.search(q):
        return RouteStrategy.TOOL_AGENT
    if _SEARCH_RE.search(q):
        return RouteStrategy.SEARCH_AGENT
    if _DIRECT_RE.search(q):
        return RouteStrategy.DIRECT_LLM
    # A bare term/entity is a grounded lookup, not chat (e.g. "FAISS").
    if _is_bare_lookup(q):
        return RouteStrategy.SEARCH_AGENT
    # No dominant signal → grounded RAG.
    return RouteStrategy.AGENTIC_RAG


_ROUTE_PROMPT = (
    "Classify how to best answer the user's request. Reply with exactly one "
    "label and nothing else:\n"
    "- direct_llm: a self-contained generative or conversational request that "
    "names no entity to look up (e.g. write a poem, translate this, say hello). "
    "Do NOT use this for a question about a named tool, product, library, "
    "person, term, or concept.\n"
    "- agentic_rag: a descriptive question best answered from the knowledge base "
    "with synthesis (e.g. summaries, comparisons, how-tos over internal docs)\n"
    "- search_agent: look up facts about a specific entity/term or current "
    "information — including a bare keyword or product/library name "
    "(e.g. 'FAISS', 'vector database benchmarks', find/look up X)\n"
    "- tool_agent: take an action via a tool or API "
    "(e.g. send, create, schedule, call an API)\n\n"
    "Request: {user_query}\n"
    "Label:"
)

_LABEL_BY_VALUE = {s.value: s for s in RouteStrategy}


def classify_route(query: str, llm: "LLMClient") -> RouteStrategy:
    """LLM-backed 4-way route classification.

    Single completion that returns one label; defaults to AGENTIC_RAG on an
    empty or unexpected response (grounded is the safe fallback).
    """
    from src.context.models import ChatMessage

    prompt = _ROUTE_PROMPT.format(user_query=query)
    response = llm.complete([ChatMessage(role="user", content=prompt)])
    content = (
        (response if isinstance(response, str) else response.content).strip().lower()
    )
    if not content:
        logger.warning("Route classification empty; defaulting to agentic_rag.")
        return RouteStrategy.AGENTIC_RAG
    for value, strategy in _LABEL_BY_VALUE.items():
        if value in content:
            return strategy
    logger.warning(
        "Route classification returned unexpected response %r; defaulting to "
        "agentic_rag.",
        content,
    )
    return RouteStrategy.AGENTIC_RAG


def route_query(
    query: str,
    *,
    llm: "LLMClient | None",
    has_local_model: bool,
    explicit_source: bool,
) -> RouteStrategy:
    """Decide the agent strategy for an auto-routed (mode=None) request.

    Cascade:
      1. An explicit non-default source provider is a search command.
      2. A bare term/entity lookup (e.g. "FAISS") is a grounded search — decided
         deterministically so it never reaches the classifier, which tends to
         over-route such lookups to direct_llm (ungrounded).
      3. With an LLM, use the 4-way classifier (rule-based on error).
      4. Without an LLM, use the rule-based route.

    ``has_local_model`` is accepted so callers can reason about capability, but
    capability-aware *degradation* happens at dispatch time, not here — this
    function returns the ideal strategy for the query.
    """
    del has_local_model  # dispatch layer handles capability degradation
    if explicit_source:
        return RouteStrategy.SEARCH_AGENT
    if _is_bare_lookup(query):
        return RouteStrategy.SEARCH_AGENT
    if llm is not None:
        try:
            return classify_route(query, llm)
        except Exception as exc:  # noqa: BLE001 — fall back, never fail routing
            logger.warning("Route classifier failed, using rule-based: %s", exc)
            return _rule_based_route(query)
    return _rule_based_route(query)
