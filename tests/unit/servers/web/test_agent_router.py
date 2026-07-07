"""Unit tests for the 3-way agentic router decision logic."""

from __future__ import annotations

import pytest

import src.internal.servers.web.intent_routing as ir
from src.context.models import ChatMessage
from src.internal.servers.web.intent_routing import (
    RouteStrategy,
    _regex_route,
    _rule_based_route,
    classify_route,
    route_query,
)


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[list[ChatMessage]] = []
        self.call_kwargs: list[dict] = []

    def complete(self, messages: list[ChatMessage], **kwargs) -> str:
        self.calls.append(messages)
        self.call_kwargs.append(kwargs)
        return self._reply


def test_classify_route_uses_deterministic_decoding():
    # The strategy classifier must decode deterministically (temperature 0), so
    # the same query always routes to the same strategy/source run-to-run.
    llm = _FakeLLM("chat")
    classify_route("compare dense and sparse retrieval", llm)

    assert llm.call_kwargs, "the classifier consulted the LLM"
    assert llm.call_kwargs[0].get("temperature") == 0.0


# --- route_query cascade ---


def test_explicit_source_routes_to_search_agent():
    # An explicit source provider is an unambiguous search command.
    strategy = route_query(
        "anything at all",
        llm=_FakeLLM("chat"),
        has_local_model=True,
        explicit_source=True,
    )
    assert strategy is RouteStrategy.SEARCH


def test_route_query_without_llm_uses_rule_based():
    # No LLM → rule-based route; a search verb yields SEARCH.
    strategy = route_query(
        "find the latest pricing sheet",
        llm=None,
        has_local_model=False,
        explicit_source=False,
    )
    assert strategy is RouteStrategy.SEARCH


def test_route_query_uses_llm_classifier_when_available():
    llm = _FakeLLM("tool")
    strategy = route_query(
        "create a Jira ticket for the outage",
        llm=llm,
        has_local_model=True,
        explicit_source=False,
    )
    assert strategy is RouteStrategy.TOOL
    assert llm.calls  # the classifier consulted the LLM


def test_route_query_bare_lookup_is_search_and_skips_classifier():
    # A bare entity/term lookup like "FAISS" is unambiguously a grounded search,
    # so it must NOT reach the (over-eager) LLM classifier that would otherwise
    # send it to chat. Deterministic regardless of the LLM reply.
    llm = _FakeLLM("chat")
    strategy = route_query(
        "FAISS", llm=llm, has_local_model=True, explicit_source=False
    )
    assert strategy is RouteStrategy.SEARCH
    assert llm.calls == []  # classifier was never consulted


def test_route_query_descriptive_phrase_still_uses_classifier():
    # A multi-word descriptive phrase is NOT a bare lookup → classifier decides.
    llm = _FakeLLM("chat")
    strategy = route_query(
        "the procurement approval flow",
        llm=llm,
        has_local_model=True,
        explicit_source=False,
    )
    assert strategy is RouteStrategy.CHAT
    assert llm.calls  # the classifier was consulted


# --- _rule_based_route ---


def test_rule_based_bare_lookup_routes_to_search():
    assert _rule_based_route("FAISS") is RouteStrategy.SEARCH
    assert _rule_based_route("vector database") is RouteStrategy.SEARCH


def test_rule_based_action_routes_to_tool():
    assert _rule_based_route("send an email to the team") is RouteStrategy.TOOL
    assert _rule_based_route("create a ticket for this bug") is RouteStrategy.TOOL


def test_rule_based_search_verb_routes_to_search():
    assert _rule_based_route("find the Q3 revenue report") is RouteStrategy.SEARCH
    assert _rule_based_route("look up the latest release notes") is RouteStrategy.SEARCH


def test_rule_based_generative_routes_to_chat():
    assert _rule_based_route("write a haiku about the sea") is RouteStrategy.CHAT
    assert _rule_based_route("translate this sentence to French") is RouteStrategy.CHAT


def test_rule_based_default_no_signal_routes_to_chat():
    assert _rule_based_route("the procurement approval flow") is RouteStrategy.CHAT


# --- classify_route ---


def test_classify_route_parses_each_label():
    for label, expected in [
        ("chat", RouteStrategy.CHAT),
        ("search", RouteStrategy.SEARCH),
        ("tool", RouteStrategy.TOOL),
    ]:
        assert classify_route("q", _FakeLLM(label))[0] is expected


def test_classify_route_defaults_to_chat_on_garbage():
    assert classify_route("q", _FakeLLM("nonsense reply"))[0] is RouteStrategy.CHAT
    assert classify_route("q", _FakeLLM(""))[0] is RouteStrategy.CHAT


def test_bare_lookup_excludes_greetings_and_generative():
    from src.internal.servers.web.intent_routing import _is_bare_lookup

    assert _is_bare_lookup("hello") is False
    assert _is_bare_lookup("poem") is False
    assert _is_bare_lookup("translate this") is False
    # A genuine entity lookup is still a bare lookup.
    assert _is_bare_lookup("FAISS") is True


def test_route_query_greeting_routes_to_chat_without_llm():
    # No LLM → rule-based; a bare greeting must NOT short-circuit to SEARCH.
    strategy = route_query(
        "hello", llm=None, has_local_model=False, explicit_source=False
    )
    assert strategy is RouteStrategy.CHAT


def test_classify_route_ignores_substring_false_positives():
    # Word-boundary match: "research" must not count as the "search" label.
    assert classify_route("q", _FakeLLM("researching options"))[0] is RouteStrategy.CHAT
    assert classify_route("q", _FakeLLM("chatbot style"))[0] is RouteStrategy.CHAT
    # Exact labels still parse.
    assert classify_route("q", _FakeLLM("search"))[0] is RouteStrategy.SEARCH


# --- _regex_route (deterministic pre-LLM pass) ---


@pytest.mark.parametrize(
    "query,expected",
    [
        # TOOL — unambiguous imperative at the start
        ("send an email to Bob", RouteStrategy.TOOL),
        ("schedule a meeting for Friday", RouteStrategy.TOOL),
        # TOOL — ambiguous verb, but object-qualified
        ("create a ticket for the outage", RouteStrategy.TOOL),
        ("open an issue about the crash", RouteStrategy.TOOL),
        # SEARCH — bare term / lookup imperative
        ("FAISS", RouteStrategy.SEARCH),
        ("find the Q3 revenue report", RouteStrategy.SEARCH),
        ("look up the release notes", RouteStrategy.SEARCH),
        # CHAT — question / explain / generative / trailing '?'
        ("What is FAISS?", RouteStrategy.CHAT),
        ("explain how to send an email", RouteStrategy.CHAT),
        ("write a haiku about the sea", RouteStrategy.CHAT),
        ("is this thing on?", RouteStrategy.CHAT),
        # None — currency conflict on a chat-form question → defer to LLM
        ("what is the latest price of NVDA", None),
        # None — no confident signal → defer to LLM
        ("the procurement approval flow", None),
        ("", None),
        # SEARCH — a lookup verb wins over both the trailing '?' and the
        # currency cue (the currency short-circuit only applies to the CHAT
        # branch, not SEARCH).
        ("find the latest report?", RouteStrategy.SEARCH),
    ],
)
def test_regex_route(query, expected):
    assert _regex_route(query) is expected


def test_regex_route_tool_verb_needs_object_when_ambiguous():
    # A bare ambiguous verb must NOT misfire to TOOL without an object.
    assert _regex_route("open source models") is RouteStrategy.SEARCH
    assert _regex_route("post office hours") is RouteStrategy.SEARCH


def test_regex_route_polysemous_verbs_are_not_bare_tool_actions():
    # These verbs are common leading nouns/adjectives, not just imperatives,
    # so they must NOT short-circuit to TOOL without an object qualifier.
    assert (
        _regex_route("book recommendations for machine learning")
        is not RouteStrategy.TOOL
    )
    assert _regex_route("email templates for onboarding") is not RouteStrategy.TOOL
    assert _regex_route("schedule of the world cup") is not RouteStrategy.TOOL
    assert _regex_route("cancel culture explained") is not RouteStrategy.TOOL
    assert _regex_route("trigger warnings in modern media") is not RouteStrategy.TOOL
    # True positives must be preserved: unambiguous bare verb, and
    # object-qualified polysemous verb.
    assert _regex_route("send an email to Bob") is RouteStrategy.TOOL
    assert _regex_route("schedule a meeting for Friday") is RouteStrategy.TOOL


# --- route_query uses _regex_route before the LLM ---


def test_route_query_confident_regex_skips_llm():
    # A confident chat-form question routes deterministically; the LLM classifier
    # is never consulted (previously this misrouted via the classifier).
    llm = _FakeLLM("search")  # would say search if consulted
    strategy = route_query(
        "What is FAISS?", llm=llm, has_local_model=True, explicit_source=False
    )
    assert strategy is RouteStrategy.CHAT
    assert llm.calls == []  # regex decided; classifier not consulted


def test_route_query_ambiguous_falls_through_to_llm():
    # No confident regex match → the LLM classifier decides.
    llm = _FakeLLM("chat")
    strategy = route_query(
        "the procurement approval flow",
        llm=llm,
        has_local_model=True,
        explicit_source=False,
    )
    assert strategy is RouteStrategy.CHAT
    assert llm.calls  # classifier consulted


def test_route_query_currency_conflict_defers_to_llm():
    # A chat-form question with a currency cue is NOT decided by regex.
    llm = _FakeLLM("search")
    strategy = route_query(
        "what is the latest price of NVDA",
        llm=llm,
        has_local_model=True,
        explicit_source=False,
    )
    assert strategy is RouteStrategy.SEARCH
    assert llm.calls  # deferred to the classifier


class _SpyLLM:
    def __init__(self):
        self.called = False

    def complete(self, *_a, **_k):
        self.called = True
        return "chat"


def test_route_query_uses_model_when_confident(monkeypatch):
    monkeypatch.setattr(ir, "predict_route", lambda q: (RouteStrategy.SEARCH, 0.9))
    llm = _SpyLLM()
    strategy = ir.route_query(
        "the vendor contract renewal terms",
        llm=llm,
        has_local_model=True,
        explicit_source=False,
    )
    assert strategy is RouteStrategy.SEARCH
    assert llm.called is False  # model replaced the LLM step


def test_route_query_defers_to_llm_when_model_low_confidence(monkeypatch):
    monkeypatch.setattr(ir, "predict_route", lambda q: (RouteStrategy.SEARCH, 0.3))
    llm = _SpyLLM()
    strategy = ir.route_query(
        "the vendor contract renewal terms",
        llm=llm,
        has_local_model=True,
        explicit_source=False,
    )
    assert llm.called is True  # low confidence -> LLM fallback
    assert strategy is RouteStrategy.CHAT  # spy LLM returns "chat"


def test_route_query_no_model_is_unchanged(monkeypatch):
    monkeypatch.setattr(ir, "predict_route", lambda q: None)
    llm = _SpyLLM()
    strategy = ir.route_query(
        "the vendor contract renewal terms",
        llm=llm,
        has_local_model=True,
        explicit_source=False,
    )
    assert llm.called is True  # no model -> today's behavior
    assert strategy is RouteStrategy.CHAT


def test_regex_still_wins_over_model(monkeypatch):
    called = {"model": False}

    def _spy(_q):
        called["model"] = True
        return (RouteStrategy.CHAT, 0.99)

    monkeypatch.setattr(ir, "predict_route", _spy)
    # "find X" matches the anchored search regex -> returns before predict_route.
    strategy = ir.route_query(
        "find the Q3 revenue report",
        llm=None,
        has_local_model=False,
        explicit_source=False,
    )
    assert strategy is RouteStrategy.SEARCH
    assert called["model"] is False
