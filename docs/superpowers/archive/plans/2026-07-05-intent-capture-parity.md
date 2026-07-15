# Intent-capture Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit exactly one `intent` capture stage per request from `route_query` (covering explicit/regex/classifier/rule_based paths), so the Request Inspector shows the routing decision for regex-decided queries too.

**Architecture:** `classify_route` returns `(strategy, {prompt, raw_label})` instead of emitting its own capture stage. `route_query` calls a `_record_intent(mechanism, strategy, detail)` helper on every decision path. `_regex_route` is untouched. Observability-only — no routing decisions change.

**Tech Stack:** Python 3.12, pytest. Single file: `src/internal/servers/web/intent_routing.py` + its tests.

## Global Constraints

- One `intent` stage per request, emitted from `route_query`, labeled by mechanism (`explicit_source | regex | classifier | rule_based`); payload `{"mechanism", "strategy", **detail}`.
- The classifier path preserves `prompt` + `raw_label` in its detail; other paths use `{}`.
- `_regex_route` keeps its pure `-> RouteStrategy | None` signature (untouched).
- `record_stage` is a no-op when no capture is active — the hot path must stay unaffected when debug panels are off.
- No routing-decision changes; no frontend change (RequestInspector renders any payload).
- Run `ruff check <files> --fix && ruff format <files>` before committing (ruff pre-commit hook; if a commit aborts because the hook reformatted, `git add -A` and re-run the same commit).
- Branch: `feat/intent-capture-parity` (spec already committed there).

---

### Task 1: Centralize intent capture in `route_query`

**Files:**
- Modify: `src/internal/servers/web/intent_routing.py` (`classify_route`; add `_record_intent`; `route_query`)
- Test: `tests/unit/servers/web/test_stage_emits_intent.py` (rewrite), `tests/unit/servers/web/test_agent_router.py` (update `test_classify_route_*` to unpack the tuple)

**Interfaces:**
- Consumes: existing `_regex_route`, `_rule_based_route`, `_capture.record_stage`, `RouteStrategy`, `_FakeLLM` (in `test_agent_router.py`).
- Produces: `classify_route(query, llm) -> tuple[RouteStrategy, dict]`; `_record_intent(mechanism: str, strategy: RouteStrategy, detail: dict) -> None`.

- [ ] **Step 1: Rewrite `test_stage_emits_intent.py` to drive `route_query` (failing)**

Replace the entire contents of `tests/unit/servers/web/test_stage_emits_intent.py` with:

```python
from __future__ import annotations

from src.context.models import ChatMessage
from src.internal.servers.web import request_capture as rc
from src.internal.servers.web.intent_routing import route_query


class _FakeLLM:
    def complete(self, messages: list[ChatMessage], **_) -> str:
        return "search"


def _intent_stages():
    return [s for s in rc.active().stages if s.stage == "intent"]


def test_route_query_emits_regex_intent_stage():
    token = rc.start_capture("r", "What is FAISS?")
    try:
        strategy = route_query(
            "What is FAISS?", llm=_FakeLLM(), has_local_model=True, explicit_source=False
        )
        stages = _intent_stages()
        assert len(stages) == 1
        assert stages[0].label == "regex"
        assert stages[0].payload["mechanism"] == "regex"
        assert stages[0].payload["strategy"] == strategy.value  # "chat"
    finally:
        rc.reset_capture(token)


def test_route_query_emits_classifier_intent_stage_with_detail():
    # A phrase _regex_route defers on → classifier path, preserving prompt/raw_label.
    token = rc.start_capture("r", "the procurement approval flow")
    try:
        route_query(
            "the procurement approval flow",
            llm=_FakeLLM(),
            has_local_model=True,
            explicit_source=False,
        )
        stages = _intent_stages()
        assert len(stages) == 1
        assert stages[0].label == "classifier"
        assert stages[0].payload["mechanism"] == "classifier"
        assert stages[0].payload["raw_label"] == "search"
        assert "prompt" in stages[0].payload
    finally:
        rc.reset_capture(token)


def test_route_query_emits_explicit_source_intent_stage():
    token = rc.start_capture("r", "anything at all")
    try:
        route_query(
            "anything at all", llm=None, has_local_model=False, explicit_source=True
        )
        stages = _intent_stages()
        assert len(stages) == 1
        assert stages[0].payload["mechanism"] == "explicit_source"
        assert stages[0].payload["strategy"] == "search"
    finally:
        rc.reset_capture(token)


def test_route_query_emits_rule_based_intent_stage_without_llm():
    token = rc.start_capture("r", "the procurement approval flow")
    try:
        route_query(
            "the procurement approval flow",
            llm=None,
            has_local_model=False,
            explicit_source=False,
        )
        stages = _intent_stages()
        assert len(stages) == 1
        assert stages[0].payload["mechanism"] == "rule_based"
    finally:
        rc.reset_capture(token)


def test_route_query_no_capture_does_not_raise():
    # With no active capture the emit is a silent no-op.
    route_query("q", llm=_FakeLLM(), has_local_model=True, explicit_source=False)
```

- [ ] **Step 2: Update `test_classify_route_*` in `test_agent_router.py` to unpack the tuple (failing)**

In `tests/unit/servers/web/test_agent_router.py`, update the three tests that assert on `classify_route`'s return value to unpack `[0]`:

`test_classify_route_parses_each_label`:
```python
def test_classify_route_parses_each_label():
    for label, expected in [
        ("chat", RouteStrategy.CHAT),
        ("search", RouteStrategy.SEARCH),
        ("tool", RouteStrategy.TOOL),
    ]:
        assert classify_route("q", _FakeLLM(label))[0] is expected
```

`test_classify_route_defaults_to_chat_on_garbage`:
```python
def test_classify_route_defaults_to_chat_on_garbage():
    assert classify_route("q", _FakeLLM("nonsense reply"))[0] is RouteStrategy.CHAT
    assert classify_route("q", _FakeLLM(""))[0] is RouteStrategy.CHAT
```

`test_classify_route_ignores_substring_false_positives`:
```python
def test_classify_route_ignores_substring_false_positives():
    # Word-boundary match: "research" must not count as the "search" label.
    assert classify_route("q", _FakeLLM("researching options"))[0] is RouteStrategy.CHAT
    assert classify_route("q", _FakeLLM("chatbot style"))[0] is RouteStrategy.CHAT
    # Exact labels still parse.
    assert classify_route("q", _FakeLLM("search"))[0] is RouteStrategy.SEARCH
```

(Leave `test_classify_route_uses_deterministic_decoding` unchanged — it discards the return value and only checks `llm.call_kwargs`.)

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/servers/web/test_stage_emits_intent.py tests/unit/servers/web/test_agent_router.py -k "route_query_emits or classify_route" -v`
Expected: FAIL — the new `route_query_emits_*` tests fail (no `intent` stage emitted from `route_query`; classifier stage still labeled `classify_route`), and the unpacked `classify_route` assertions fail (`classify_route` still returns a bare `RouteStrategy`, so `[0]` indexes the enum wrongly / raises).

- [ ] **Step 4: Change `classify_route` to return a tuple and stop emitting**

In `src/internal/servers/web/intent_routing.py`, change `classify_route`'s signature and its tail. Replace the `record_stage(...)` call + `return strategy` at the end with a tuple return, and update the signature/docstring:

```python
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
```

- [ ] **Step 5: Add `_record_intent` and emit from every `route_query` path**

Immediately before `def route_query(` in the same file, add:

```python
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
```

Then rewrite the body of `route_query` (from `del has_local_model` to the end) to emit on each path:

```python
    del has_local_model  # dispatch layer handles capability degradation
    if explicit_source:
        _record_intent("explicit_source", RouteStrategy.SEARCH, {})
        return RouteStrategy.SEARCH
    regex_choice = _regex_route(query)
    if regex_choice is not None:
        _record_intent("regex", regex_choice, {})
        return regex_choice
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
```

- [ ] **Step 6: Run the intent + router tests to verify they pass**

Run: `python -m pytest tests/unit/servers/web/test_stage_emits_intent.py tests/unit/servers/web/test_agent_router.py -v`
Expected: PASS — the five `route_query_emits_*` tests pass, all `classify_route` tests pass with the tuple unpack, and every pre-existing router test still passes (routing decisions unchanged).

- [ ] **Step 7: Run the web suite for regressions**

Run: `python -m pytest tests/unit/servers/web/ -q`
Expected: PASS — including `test_request_capture_e2e.py` (its query defers to the classifier, which still records an `intent` stage) and the capture-module tests (which use `"classify_route"` only as an arbitrary literal label, unaffected).

- [ ] **Step 8: Commit**

```bash
ruff check src/internal/servers/web/intent_routing.py tests/unit/servers/web/test_stage_emits_intent.py tests/unit/servers/web/test_agent_router.py --fix && ruff format src/internal/servers/web/intent_routing.py tests/unit/servers/web/test_stage_emits_intent.py tests/unit/servers/web/test_agent_router.py
git add src/internal/servers/web/intent_routing.py tests/unit/servers/web/test_stage_emits_intent.py tests/unit/servers/web/test_agent_router.py
git commit -m "feat(devconsole): centralize intent capture in route_query (all mechanisms)"
```

---

## Self-Review

**Spec coverage:** centralized emit in `route_query` with `_record_intent` (all four mechanisms) → Steps 5. `classify_route` returns `(strategy, {prompt, raw_label})` and stops emitting → Step 4. `_regex_route` untouched → not modified anywhere. No-op when inactive → `record_stage` behavior, exercised by `test_route_query_no_capture_does_not_raise`. Test updates (classify_route unpack, repoint intent-emit test, per-mechanism capture tests, e2e stays green) → Steps 1, 2, 6, 7. All spec sections covered.

**Placeholder scan:** every step has concrete code, exact paths, exact commands, expected output. No TBD/TODO.

**Type consistency:** `classify_route(query, llm) -> tuple[RouteStrategy, dict]` (Step 4) is unpacked as `strategy, detail = classify_route(...)` in `route_query` (Step 5) and `[0]` in the updated tests (Step 2). `_record_intent(mechanism: str, strategy: RouteStrategy, detail: dict)` (Step 5) is called with those exact argument types on every path. Stage `label` == `mechanism`; tests assert `stages[0].label`/`payload["mechanism"]` consistently. `_FakeLLM` reused from `test_agent_router.py`; a local `_FakeLLM` returning `"search"` is defined in the rewritten `test_stage_emits_intent.py`.
