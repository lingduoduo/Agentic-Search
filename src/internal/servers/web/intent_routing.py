"""Intent routing helpers for the /api/agent endpoint."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.base import AgentLoopOutput

_TEMPORAL_RE = re.compile(
    r"\b(today|right now|this week|this month|this year|latest|breaking|news|recent|"
    r"currently|202[4-9]|203\d)\b",
    re.IGNORECASE,
)

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


def _rule_based_is_search(query: str) -> bool:
    """Return True if the query looks like a search/retrieval intent."""
    q = query.strip()
    if not q:
        return False
    # Check for explicit chat keywords first
    if _CHAT_RE.search(q):
        return False
    # Check for explicit search keywords
    if _SEARCH_RE.search(q):
        return True
    # Short queries without verbs are treated as search (e.g., "procurement process")
    tokens = q.split()
    if len(tokens) <= 5 and not _VERB_RE.search(q) and not q.endswith("?"):
        return True
    return False


def _route_source_provider(query: str, browser_search_url: str | None = None) -> str:
    """Pick the retrieval backend for a search query.

    Temporal queries ("today", "latest news", year references) route to the
    browser backend when one is configured.  Everything else uses the local
    corpus retrieval server.
    """
    if browser_search_url and _TEMPORAL_RE.search(query):
        return "browser"
    return "retrieval"


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
