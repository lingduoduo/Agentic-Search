"""Intent routing helpers for the /api/agent endpoint."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from src.internal.servers.web import request_capture as _capture

if TYPE_CHECKING:
    from src.agents.core.base import AgentLoopOutput
    from src.context.models import LLMClient
    from src.internal.configs import AppSettings

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
        if tool_name == "search":
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


@dataclass(frozen=True)
class ClarificationOption:
    """One route the user can choose when the router could not decide."""

    route: str
    label: str


@dataclass(frozen=True)
class Clarification:
    """A question to ask instead of guessing a route."""

    question: str
    options: tuple[ClarificationOption, ...]


@dataclass(frozen=True)
class RouteDecision:
    """The chosen route, plus a question when that choice was a guess.

    ``strategy`` is always set and always equals what the pre-clarification
    cascade returned, so ``route_query`` stays behavior-identical.
    """

    strategy: RouteStrategy
    clarification: "Clarification | None" = None


_CLARIFICATION = Clarification(
    question=("I can take this a few different ways — which would you like?"),
    options=(
        ClarificationOption("chat", "Explain or summarize it"),
        ClarificationOption("search", "Find the document or facts"),
        ClarificationOption("tool", "Take an action on it"),
    ),
)


# Imported after RouteStrategy is defined: ml_intent imports RouteStrategy from
# this module at its own top level, so importing ml_intent any earlier here
# would hit a circular-import error (RouteStrategy not yet defined).
from src.internal.servers.web.ml_intent import predict_route  # noqa: E402


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


def _rule_based_route_or_none(query: str) -> "RouteStrategy | None":
    """Heuristic 3-way route, or None when no cue dominates."""
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
    return None


def _rule_based_route(query: str) -> RouteStrategy:
    """Heuristic 3-way route. Precedence: tool > search > bare-lookup > chat.

    The default is CHAT: when no signal dominates, a grounded answer is safer
    than an ungrounded one.
    """
    return _rule_based_route_or_none(query) or RouteStrategy.CHAT


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


def classify_route(query: str, llm: "LLMClient") -> "tuple[RouteStrategy | None, dict]":
    """LLM-backed 3-way route classification.

    Returns ``(strategy, detail)`` where ``detail`` is
    ``{"raw_label": ...}``, where ``raw_label`` is restricted to a supported
    label or the ``empty``/``unexpected`` sentinel. The prompt and arbitrary
    completion text are deliberately omitted because either can contain the raw
    user query and this detail is recorded in request captures. Returns
    ``strategy=None`` on an empty or unexpected response; the caller supplies
    the ``chat`` default. The caller (``route_query``/``route_request``)
    records the intent capture stage, so this no longer emits one itself.
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
    strategy: "RouteStrategy | None" = None
    captured_label = "empty"
    if not content:
        logger.warning("Route classification empty; defaulting to chat.")
    else:
        for value, mapped in _LABEL_BY_VALUE.items():
            if re.search(rf"\b{value}\b", content):
                strategy = mapped
                captured_label = value
                break
        else:
            captured_label = "unexpected"
            logger.warning("Route classification response invalid; defaulting to chat.")
    return strategy, {"raw_label": captured_label}


def _record_intent(mechanism: str, strategy: RouteStrategy, detail: dict) -> None:
    """Record the single intent capture stage for the chosen route.

    No-op when no capture is active. Labeled by ``mechanism`` so the Request
    Inspector shows ``intent · rules`` vs ``intent · classifier``, etc.
    """
    _capture.record_stage(
        "intent",
        mechanism,
        {"mechanism": mechanism, "strategy": strategy.value, **detail},
    )


def route_request(
    query: str,
    *,
    llm: "LLMClient | None",
    explicit_source: bool,
    settings: "AppSettings | None" = None,
    telemetry: dict | None = None,
) -> RouteDecision:
    """Decide the agent strategy for an auto-routed (mode=None) request, or ask.

    When *telemetry* is supplied it is filled in with the deciding mechanism and
    any model evaluation, so callers can persist route outcomes in production.
    Request captures only run under the debug panels, and the query itself is
    already stored with the session, so nothing new about the request is logged.

    Cascade:
      1. An explicit non-default source provider is a search command.
      2. A confident `_regex_route` match (anchored tool/search/chat cues,
         incl. bare lookup) is returned deterministically, skipping the
         classifier.
      3. A similarity match against the curated canonical examples
         (`predict_route`) whose cosine confidence clears its typed serving
         threshold is returned, replacing the LLM step. Nothing is trained:
         the route is the one whose nearest canonical examples are closest.
      4. With an LLM, use the 3-way classifier (rule-based on error).
      5. Without an LLM, use the rule-based route.

    Capability-aware *degradation* happens at dispatch time, not here — this
    function returns the ideal strategy for the query.

    Two guess-sites fall through with no signal at all: an unusable classifier
    response, and a heuristic with no dominant cue. Both return CHAT (today's
    default) plus a `Clarification`, unless `settings.route_clarification` is
    off, in which case the CHAT default is returned unadorned.
    """

    def decided(mechanism: str, strategy: RouteStrategy, detail: dict) -> RouteDecision:
        _record_intent(mechanism, strategy, detail)
        if telemetry is not None:
            telemetry["route_mechanism"] = mechanism
        return RouteDecision(strategy)

    def guessed(strategy: RouteStrategy, detail: dict) -> RouteDecision:
        # A guess with no signal at all: "clarify" when the question is
        # actually asked, or "heuristic_default" (nothing else worked) when
        # settings turn clarification off and the CHAT default is returned
        # unadorned.
        if settings is not None and not settings.route_clarification:
            return decided("heuristic_default", strategy, detail)
        decision = decided("clarify", strategy, detail)
        return RouteDecision(decision.strategy, _CLARIFICATION)

    if explicit_source:
        return decided("explicit_source", RouteStrategy.SEARCH, {})
    regex_choice = _regex_route(query)
    if regex_choice is not None:
        return decided("rules", regex_choice, {})
    fallback_detail: dict = {}
    model_choice = predict_route(query, settings=settings)
    if model_choice is not None:
        # Two independent abstentions reach here. The margin gate reports
        # itself on the decision; the confidence gate is derived here, as it
        # always has been. Either one defers to the classifier below.
        fallback_reason = model_choice.abstain_reason or (
            "model_below_threshold"
            if model_choice.confidence < model_choice.threshold
            else None
        )
        abstained = fallback_reason is not None
        model_detail = {
            "predicted_intent": model_choice.strategy.value,
            "confidence": model_choice.confidence,
            "threshold": model_choice.threshold,
            "margin": model_choice.margin,
            "abstained": abstained,
            "fallback_reason": fallback_reason,
            "latency_ms": model_choice.latency_ms,
            "modules": list(model_choice.modules),
            "composite": model_choice.composite,
        }
        _capture.record_stage("intent_model", "evaluation", model_detail)
        if telemetry is not None:
            telemetry.update(
                route_predicted_intent=model_choice.strategy.value,
                route_confidence=model_choice.confidence,
                route_threshold=model_choice.threshold,
                route_abstained=abstained,
                route_model_latency_ms=model_choice.latency_ms,
                # Persisted in production, not only under the debug panels.
                # The composite flag exists to give a future plan-aware router
                # measured data; recording it only in a dev-only capture meant
                # it gathered none.
                route_modules=list(model_choice.modules),
                route_composite=model_choice.composite,
            )
        if not abstained:
            return decided("model", model_choice.strategy, model_detail)
        fallback_detail = {"fallback_reason": fallback_reason}
    if telemetry is not None and fallback_detail:
        telemetry["route_fallback_reason"] = fallback_detail["fallback_reason"]
    if llm is not None:
        try:
            strategy, detail = classify_route(query, llm)
            merged = {**detail, **fallback_detail}
            if strategy is None:
                return guessed(RouteStrategy.CHAT, merged)
            return decided("classifier", strategy, merged)
        except Exception:  # noqa: BLE001 — fall back, never fail routing
            logger.warning("Route classifier failed, using rule-based.")
    heuristic = _rule_based_route_or_none(query)
    if heuristic is not None:
        return decided("heuristic_default", heuristic, fallback_detail)
    return guessed(RouteStrategy.CHAT, fallback_detail)


def route_query(
    query: str,
    *,
    llm: "LLMClient | None",
    explicit_source: bool,
    settings: "AppSettings | None" = None,
    telemetry: dict | None = None,
) -> RouteStrategy:
    """Legacy shim: the route the cascade picks, ignoring any clarification."""
    return route_request(
        query,
        llm=llm,
        explicit_source=explicit_source,
        settings=settings,
        telemetry=telemetry,
    ).strategy
