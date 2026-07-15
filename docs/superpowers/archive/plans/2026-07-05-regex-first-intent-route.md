# Regex-first Intent Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a high-precision, anchored regex pass that decides obvious search/chat/tool queries deterministically before the LLM classifier, deferring only ambiguous input to `classify_route`.

**Architecture:** One new pure function `_regex_route(query) -> RouteStrategy | None` in `src/internal/servers/web/intent_routing.py`, wired into `route_query` as a pre-LLM pass. It returns a strategy only on a confident, start-anchored match, else `None` (defer). `classify_route` (LLM) and `_rule_based_route` (lenient no-LLM fallback) keep their current roles.

**Tech Stack:** Python 3.12, `re`, pytest.

## Global Constraints

- `_regex_route` is high-precision: return a strategy ONLY on a confident match; return `None` on no-match or a currency/fact cross-cue conflict.
- Anchor tool/search/chat imperative cues to the START of the stripped query (`^`), so a command (`send an email`) differs from a description (`how to send an email`).
- Reuse the existing `_is_bare_lookup`; do NOT change `classify_route` or `_rule_based_route` behavior.
- No new dependencies. No change to `app.py` dispatch/degradation.
- Run `ruff check <files> --fix && ruff format <files>` before each commit (repo has a ruff pre-commit hook; if a commit aborts because the hook reformatted files, `git add -A` and re-run the same commit).
- Branch: `feat/regex-first-intent-route` (spec already committed there).

---

### Task 1: `_regex_route` pure function + unit tests

**Files:**
- Modify: `src/internal/servers/web/intent_routing.py` (add regex constants after the existing `_TOOL_RE` block ~line 86, and add `_regex_route` after `_rule_based_route` ~line 129)
- Test: `tests/unit/servers/web/test_agent_router.py` (append)

**Interfaces:**
- Consumes: existing `RouteStrategy`, `_is_bare_lookup` (same module).
- Produces: `_regex_route(query: str) -> RouteStrategy | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/servers/web/test_agent_router.py`:

```python
# --- _regex_route (deterministic pre-LLM pass) ---

import pytest
from src.internal.servers.web.intent_routing import _regex_route


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
    ],
)
def test_regex_route(query, expected):
    assert _regex_route(query) is expected


def test_regex_route_tool_verb_needs_object_when_ambiguous():
    # A bare ambiguous verb must NOT misfire to TOOL without an object.
    assert _regex_route("open source models") is not RouteStrategy.TOOL
    assert _regex_route("post office hours") is not RouteStrategy.TOOL
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/servers/web/test_agent_router.py -k regex_route -v`
Expected: FAIL with `ImportError: cannot import name '_regex_route'`

- [ ] **Step 3: Add the regex constants and function**

In `src/internal/servers/web/intent_routing.py`, after the existing `_TOOL_RE = re.compile(...)` block (~line 86), add:

```python
# --- Deterministic pre-LLM route cues (start-anchored, high precision) ---
_TOOL_ACTION_RE = re.compile(
    r"^\s*(send|email|schedule|book|cancel|deploy|assign|notify|remind|invoke|"
    r"trigger|subscribe|unsubscribe)\b",
    re.IGNORECASE,
)
_TOOL_OBJECT_RE = re.compile(
    r"^\s*(?:create|delete|remove|update|add|open|close|file|post|run|execute) "
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
```

After `_rule_based_route` (~line 129), add:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/servers/web/test_agent_router.py -k regex_route -v`
Expected: PASS (all parametrized cases + the ambiguous-verb test)

- [ ] **Step 5: Commit**

```bash
ruff check src/internal/servers/web/intent_routing.py tests/unit/servers/web/test_agent_router.py --fix && ruff format src/internal/servers/web/intent_routing.py tests/unit/servers/web/test_agent_router.py
git add src/internal/servers/web/intent_routing.py tests/unit/servers/web/test_agent_router.py
git commit -m "feat(routing): _regex_route — anchored deterministic 3-way pre-LLM pass"
```

---

### Task 2: wire `_regex_route` into `route_query`

**Files:**
- Modify: `src/internal/servers/web/intent_routing.py` (`route_query`, ~line 179)
- Test: `tests/unit/servers/web/test_agent_router.py` (append)

**Interfaces:**
- Consumes: `_regex_route` (Task 1), existing `classify_route`, `_rule_based_route`.
- Produces: no signature change to `route_query`; new behavior — confident regex short-circuits before the LLM.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/servers/web/test_agent_router.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/servers/web/test_agent_router.py -k "confident_regex or ambiguous_falls or currency_conflict" -v`
Expected: FAIL — `test_route_query_confident_regex_skips_llm` fails because the current `route_query` sends `What is FAISS?` to the classifier (`llm.calls` is non-empty and strategy is `SEARCH`).

- [ ] **Step 3: Wire `_regex_route` into `route_query`**

In `route_query` (~line 179), replace the bare-lookup short-circuit with the regex pass. Change:

```python
    del has_local_model  # dispatch layer handles capability degradation
    if explicit_source:
        return RouteStrategy.SEARCH
    if _is_bare_lookup(query):
        return RouteStrategy.SEARCH
    if llm is not None:
        try:
            return classify_route(query, llm)
        except Exception as exc:  # noqa: BLE001 — fall back, never fail routing
            logger.warning("Route classifier failed, using rule-based: %s", exc)
            return _rule_based_route(query)
    return _rule_based_route(query)
```

to:

```python
    del has_local_model  # dispatch layer handles capability degradation
    if explicit_source:
        return RouteStrategy.SEARCH
    regex_choice = _regex_route(query)
    if regex_choice is not None:
        return regex_choice
    if llm is not None:
        try:
            return classify_route(query, llm)
        except Exception as exc:  # noqa: BLE001 — fall back, never fail routing
            logger.warning("Route classifier failed, using rule-based: %s", exc)
            return _rule_based_route(query)
    return _rule_based_route(query)
```

Also update the `route_query` docstring cascade list: replace the bare-lookup bullet (item 2) with "A confident `_regex_route` match (anchored tool/search/chat cues, incl. bare lookup) is returned deterministically, skipping the classifier."

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/servers/web/test_agent_router.py -v`
Expected: PASS — the three new tests pass AND all pre-existing router tests still pass (the bare-lookup and descriptive-phrase cases behave identically: `FAISS` → SEARCH via `_regex_route`; `the procurement approval flow` → None → classifier).

- [ ] **Step 5: Run the web suite for regressions**

Run: `python -m pytest tests/unit/servers/web/ -q`
Expected: PASS (no regressions; the e2e capture test still routes its multi-word query through the classifier).

- [ ] **Step 6: Commit**

```bash
ruff check src/internal/servers/web/intent_routing.py tests/unit/servers/web/test_agent_router.py --fix && ruff format src/internal/servers/web/intent_routing.py tests/unit/servers/web/test_agent_router.py
git add src/internal/servers/web/intent_routing.py tests/unit/servers/web/test_agent_router.py
git commit -m "feat(routing): route_query runs _regex_route before the LLM classifier"
```

---

## Self-Review

**Spec coverage:** `_regex_route` function + all 5 rules (tool anchored/object-qualified, search bare+lookup, chat question/generative/trailing-?, currency conflict guard, none) → Task 1. Wiring before the LLM, classify_route/_rule_based_route unchanged → Task 2. Testing (unit per rule + integration asserting LLM-not-consulted vs consulted + regression) → Tasks 1 & 2 steps. All spec sections covered.

**Placeholder scan:** every step has concrete code, exact paths, exact commands, and expected output. No TBD/TODO.

**Type consistency:** `_regex_route(query: str) -> RouteStrategy | None` is defined in Task 1 and consumed with that exact signature in Task 2. Regex constant names (`_TOOL_ACTION_RE`, `_TOOL_OBJECT_RE`, `_SEARCH_LOOKUP_RE`, `_CHAT_START_RE`, `_GENERATIVE_START_RE`, `_CURRENCY_RE`) are used consistently. `_FakeLLM` (with `.calls`) already exists at the top of `test_agent_router.py` and is reused.
