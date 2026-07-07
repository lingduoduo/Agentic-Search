"""Intent routing helpers for the /api/agent endpoint."""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import TYPE_CHECKING

from src.internal.servers.web import request_capture as _capture

if TYPE_CHECKING:
    from src.agents.core.base import AgentLoopOutput
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
# Conversational / generative asks that need no retrieval — used only to keep
# such queries out of the bare-lookup fast path so they fall through to CHAT.
_GENERATIVE_RE = re.compile(
    r"\b(write|translate|rephrase|reword|rewrite|draft|brainstorm|"
    r"hello|hi there|thanks|joke|poem|haiku)\b",
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
# 3-way agentic routing: pick the strategy, then the web layer dispatches the
# matching agent loop (grounded chat / multi-turn search / tool use).
# ---------------------------------------------------------------------------


class RouteStrategy(str, Enum):
    """High-level agent strategy chosen by the entry-point router.

    Values match the user-facing ``intent`` vocabulary so ``extra["route"]``
    (the chosen strategy) reads the same as the surfaced ``intent`` (what
    actually ran after degradation).
    """

    CHAT = "chat"  # grounded synthesis via AgenticRAGLoop / degraded pipeline
    SEARCH = "search"  # multi-turn search until evidence suffices
    TOOL = "tool"  # OpenAPI / MCP function calling


# Imported after RouteStrategy is defined: ml_intent imports RouteStrategy from
# this module at its own top level, so importing ml_intent any earlier here
# would hit a circular-import error (RouteStrategy not yet defined).
from src.internal.servers.web.ml_intent import intent_min_confidence, predict_route  # noqa: E402


# Imperative verbs that imply taking an action through a tool/MCP.
_TOOL_RE = re.compile(
    r"\b(send|email|create|open (?:a|an) (?:ticket|issue|pr)|file (?:a|an) "
    r"(?:ticket|issue)|schedule|book|call the api|invoke|run the|execute|"
    r"post to|update the|delete the|add to)\b",
    re.IGNORECASE,
)

# --- Deterministic pre-LLM route cues (start-anchored, high precision) ---
_TOOL_ACTION_RE = re.compile(
    r"^\s*(send|deploy|assign|notify|remind|invoke|subscribe|unsubscribe)\b",
    re.IGNORECASE,
)
_TOOL_OBJECT_RE = re.compile(
    r"^\s*(?:create|delete|remove|update|add|open|close|file|post|run|execute|"
    r"book|email|schedule|cancel|trigger) "
    r"(?:a |an |the )?"
    r"(?:ticket|issue|pr|pull request|task|event|meeting|reminder|calendar|"
    r"record|entry|api|job|workflow|deployment|message|email)\b",
    re.IGNORECASE,
)
_SEARCH_LOOKUP_RE = re.compile(
    r"^\s*(find|search for|look up|look for|retrieve|fetch|pull|list|locate|"
    r"show me|get me)\b",
    re.IGNORECASE,
)
_CHAT_START_RE = re.compile(
    r"^\s*(what|why|how|explain|describe|summarize|compare|tell me about|"
    r"difference between)\b",
    re.IGNORECASE,
)
_GENERATIVE_START_RE = re.compile(
    r"^\s*(write|draft|translate|rephrase|reword|brainstorm|compose|generate)\b",
    re.IGNORECASE,
)
# A currency/fact cue turns a chat-form question into a likely search — the one
# cross-cue conflict we detect, to defer such queries to the LLM classifier.
_CURRENCY_RE = re.compile(
    r"\b(latest|current|recent|news|price|stock|weather|today|now)\b",
    re.IGNORECASE,
)


def _is_bare_lookup(query: str) -> bool:
    """True for a short, verb-less term/entity, e.g. "FAISS", "vector database".

    Such a query is unambiguously a grounded lookup, so it routes to
    SEARCH deterministically rather than risking the LLM classifier
    sending it to chat/direct answers (ungrounded). Anything carrying a tool,
    search, conversational, generative, question, or auxiliary-verb signal is
    excluded — those are handled by the normal cascade / classifier.
    """
    q = query.strip()
    if not q or q.endswith("?"):
        return False
    if (
        _TOOL_RE.search(q)
        or _SEARCH_RE.search(q)
        or _GENERATIVE_RE.search(q)
        or _CHAT_RE.search(q)
        or _VERB_RE.search(q)
    ):
        return False
    return len(q.split()) <= 3


def _rule_based_route(query: str) -> RouteStrategy:
    """Heuristic 3-way route. Precedence: tool > search > bare-lookup > chat.

    The default is CHAT: when no signal dominates, a grounded answer is safer
    than an ungrounded one.
    """
    q = query.strip()
    if not q:
        return RouteStrategy.CHAT
    if _TOOL_RE.search(q):
        return RouteStrategy.TOOL
    if _SEARCH_RE.search(q):
        return RouteStrategy.SEARCH
    # A bare term/entity is a grounded lookup, not chat (e.g. "FAISS").
    if _is_bare_lookup(q):
        return RouteStrategy.SEARCH
    # No dominant signal → grounded chat.
    return RouteStrategy.CHAT


def _regex_route(query: str) -> "RouteStrategy | None":
    """High-precision deterministic 3-way route; None when not confident.

    Cues are anchored to the START of the query so a command ('send an email')
    is distinguished from a description ('how to send an email'). Returns None
    (defer to the LLM classifier) on no match or a known currency cross-cue.
    Precedence: tool > search > chat.
    """
    q = query.strip()
    if not q:
        return None
    if _TOOL_ACTION_RE.search(q) or _TOOL_OBJECT_RE.search(q):
        return RouteStrategy.TOOL
    if _is_bare_lookup(q) or _SEARCH_LOOKUP_RE.search(q):
        return RouteStrategy.SEARCH
    if _CHAT_START_RE.search(q) or _GENERATIVE_START_RE.search(q) or q.endswith("?"):
        if _CURRENCY_RE.search(q):
            return None
        return RouteStrategy.CHAT
    return None


_ROUTE_PROMPT = (
    "Classify how to best answer the user's request. Reply with exactly one "
    "label and nothing else:\n"
    "- chat: a descriptive or conversational question best answered from the "
    "knowledge base with synthesis, or a self-contained generative request "
    "(e.g. summaries, comparisons, how-tos, write a poem, translate this)\n"
    "- search: look up facts about a specific entity/term or current "
    "information — including a bare keyword or product/library name "
    "(e.g. 'FAISS', 'vector database benchmarks', find/look up X)\n"
    "- tool: take an action via a tool or API "
    "(e.g. send, create, schedule, call an API)\n\n"
    "Request: {user_query}\n"
    "Label:"
)

_LABEL_BY_VALUE = {s.value: s for s in RouteStrategy}


def classify_route(query: str, llm: "LLMClient") -> "tuple[RouteStrategy, dict]":
    """LLM-backed 3-way route classification.

    Returns ``(strategy, detail)`` where ``detail`` is
    ``{"prompt": ..., "raw_label": ...}``. Defaults to CHAT on an empty or
    unexpected response. The caller (``route_query``) records the intent capture
    stage, so this no longer emits one itself.
    """
    from src.context.models import ChatMessage

    prompt = _ROUTE_PROMPT.format(user_query=query)
    # Deterministic decoding so the same query always routes to the same
    # strategy/source run-to-run; the server default (~1.0) would otherwise
    # let a fixed query flip between chat/search/tool across requests.
    response = llm.complete([ChatMessage(role="user", content=prompt)], temperature=0.0)
    content = (
        (response if isinstance(response, str) else response.content).strip().lower()
    )
    strategy = RouteStrategy.CHAT
    if not content:
        logger.warning("Route classification empty; defaulting to chat.")
    else:
        for value, mapped in _LABEL_BY_VALUE.items():
            if re.search(rf"\b{value}\b", content):
                strategy = mapped
                break
        else:
            logger.warning(
                "Route classification returned unexpected response %r; defaulting to chat.",
                content,
            )
    return strategy, {"prompt": prompt, "raw_label": content}


def _record_intent(mechanism: str, strategy: RouteStrategy, detail: dict) -> None:
    """Record the single intent capture stage for the chosen route.

    No-op when no capture is active. Labeled by ``mechanism`` so the Request
    Inspector shows ``intent · regex`` vs ``intent · classifier``, etc.
    """
    _capture.record_stage(
        "intent",
        mechanism,
        {"mechanism": mechanism, "strategy": strategy.value, **detail},
    )


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
      2. A confident `_regex_route` match (anchored tool/search/chat cues,
         incl. bare lookup) is returned deterministically, skipping the
         classifier.
      3. A trained intent model (`predict_route`) whose confidence clears
         `intent_min_confidence()` is returned, replacing the LLM step.
      4. With an LLM, use the 3-way classifier (rule-based on error).
      5. Without an LLM, use the rule-based route.

    ``has_local_model`` is accepted so callers can reason about capability, but
    capability-aware *degradation* happens at dispatch time, not here — this
    function returns the ideal strategy for the query.
    """
    del has_local_model  # dispatch layer handles capability degradation
    if explicit_source:
        _record_intent("explicit_source", RouteStrategy.SEARCH, {})
        return RouteStrategy.SEARCH
    regex_choice = _regex_route(query)
    if regex_choice is not None:
        _record_intent("regex", regex_choice, {})
        return regex_choice
    model_choice = predict_route(query)
    if model_choice is not None:
        strategy, confidence = model_choice
        if confidence >= intent_min_confidence():
            _record_intent("model", strategy, {"confidence": confidence})
            return strategy
    if llm is not None:
        try:
            strategy, detail = classify_route(query, llm)
            _record_intent("classifier", strategy, detail)
            return strategy
        except Exception as exc:  # noqa: BLE001 — fall back, never fail routing
            logger.warning("Route classifier failed, using rule-based: %s", exc)
            strategy = _rule_based_route(query)
            _record_intent("rule_based", strategy, {})
            return strategy
    strategy = _rule_based_route(query)
    _record_intent("rule_based", strategy, {})
    return strategy
