# Route Clarification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the intent router say "I don't know" and ask the user, instead of silently presenting a guess as a decision.

**Architecture:** The cascade in `src/internal/servers/web/intent_routing.py` gains a `route_request` entry point returning a `RouteDecision` that carries both the strategy it would pick and, when that pick was a guess, a `Clarification`. `route_query` becomes a one-line shim over it and keeps its exact behavior. The web layer short-circuits a clarification into a response that runs no agent; the user's pick returns as a new `route` request field that re-enters the same auto dispatcher.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, pytest, React 19 + TypeScript (Vite), scikit-learn (offline evaluation only).

## Global Constraints

- Preserve `INTENT_LABELS: list[str] = ["chat", "search", "tool"]` in that order.
- Do not add a member to `RouteStrategy`. `ml_intent._ROUTE_VALUES` and `intent_routing._LABEL_BY_VALUE` derive from it and would silently widen.
- Every request that reaches a signal today keeps its exact route and agent.
- Explicit modes, explicit sources, deterministic rules, and confident model predictions never clarify.
- Clarification text is static. Never call an LLM to build it.
- Do not log raw request text for intent telemetry.
- Do not change the tool→chat degradation or the `/chat`, `/search`, `/tools` endpoint stacks.
- Do not add a new runtime dependency.
- `AGENTIC_SEARCH_ROUTE_CLARIFICATION` defaults to `true`.

## File Structure

- Modify `src/internal/servers/web/intent_routing.py`: `Clarification`, `ClarificationOption`, `RouteDecision`, `route_request`, guess-site detection, renamed mechanism labels.
- Modify `src/internal/configs/app_configs.py` and `src/internal/configs/default_config.py`: the clarification toggle.
- Modify `src/internal/servers/web/app.py`: `route` request field, clarification short-circuit, response and SSE fields.
- Modify `src/model/intent_evaluation.py`: model-scoped tool precision.
- Modify `web/src/types.ts` and `web/src/pages/AssistPage.tsx`: render the choices and re-submit.
- Modify `docs/request-routing.md`, `docs/configuration.md`, `docs/training-and-evaluation.md`.

---

### Task 1: Guess-site detection and the RouteDecision core

**Files:**
- Modify: `src/internal/servers/web/intent_routing.py`
- Modify: `src/internal/configs/app_configs.py`
- Modify: `src/internal/configs/default_config.py`
- Modify: `tests/unit/servers/web/test_agent_router.py`
- Modify: `tests/unit/test_configs.py`

**Interfaces:**
- Consumes: `RouteStrategy`, `_regex_route`, `_rule_based_route`, `classify_route`, `predict_route`, `AppSettings`.
- Produces: `ClarificationOption(route: str, label: str)`.
- Produces: `Clarification(question: str, options: tuple[ClarificationOption, ...])`.
- Produces: `RouteDecision(strategy: RouteStrategy, clarification: Clarification | None = None)`.
- Produces: `route_request(query, *, llm, explicit_source, settings=None, telemetry=None) -> RouteDecision`.
- Produces: `AppSettings.route_clarification: bool` (default `True`).
- Preserves: `route_query(query, *, llm, explicit_source, settings=None, telemetry=None) -> RouteStrategy`.

- [ ] **Step 1: Write the failing guess-site tests**

Add to `tests/unit/servers/web/test_agent_router.py`:

```python
def test_route_request_clarifies_when_heuristic_has_no_signal():
    decision = ir.route_request(
        "Review the vendor renewal terms", llm=None, explicit_source=False
    )

    assert decision.strategy is RouteStrategy.CHAT  # legacy answer preserved
    assert decision.clarification is not None
    assert [option.route for option in decision.clarification.options] == [
        "chat",
        "search",
        "tool",
    ]


def test_route_request_does_not_clarify_when_heuristic_matches_a_cue():
    decision = ir.route_request(
        "email the quarterly report to legal", llm=None, explicit_source=False
    )

    assert decision.strategy is RouteStrategy.TOOL
    assert decision.clarification is None


def test_route_request_clarifies_on_unusable_llm_output():
    class _GarbageLLM:
        def complete(self, messages, **kwargs):
            return "I'm not sure what you mean"

    decision = ir.route_request(
        "Review the vendor renewal terms",
        llm=_GarbageLLM(),
        explicit_source=False,
    )

    assert decision.strategy is RouteStrategy.CHAT  # today's classifier default
    assert decision.clarification is not None


def test_route_request_never_clarifies_on_a_deterministic_rule():
    for query in ("find the onboarding checklist", "Explain how FAISS works"):
        decision = ir.route_request(query, llm=None, explicit_source=False)
        assert decision.clarification is None


def test_route_request_does_not_clarify_on_a_usable_llm_label():
    class _UsableLLM:
        def complete(self, messages, **kwargs):
            return "search"

    decision = ir.route_request(
        "Review the vendor renewal terms", llm=_UsableLLM(), explicit_source=False
    )

    assert decision.strategy is RouteStrategy.SEARCH
    assert decision.clarification is None


def test_route_request_falls_through_to_the_heuristic_when_the_llm_raises():
    class _BrokenLLM:
        def complete(self, messages, **kwargs):
            raise RuntimeError("provider down")

    decision = ir.route_request(
        "email the quarterly report to legal",
        llm=_BrokenLLM(),
        explicit_source=False,
    )

    assert decision.strategy is RouteStrategy.TOOL
    assert decision.clarification is None


def test_route_request_never_clarifies_on_a_confident_model(monkeypatch):
    monkeypatch.setattr(
        ir,
        "predict_route",
        lambda query, settings=None: IntentModelDecision(
            strategy=RouteStrategy.TOOL,
            confidence=0.91,
            threshold=0.6,
            latency_ms=1.5,
        ),
    )

    decision = ir.route_request(
        "Review the vendor renewal terms", llm=None, explicit_source=False
    )

    assert decision.strategy is RouteStrategy.TOOL
    assert decision.clarification is None


def test_route_request_never_clarifies_for_an_explicit_source():
    decision = ir.route_request(
        "Review the vendor renewal terms", llm=None, explicit_source=True
    )

    assert decision.strategy is RouteStrategy.SEARCH
    assert decision.clarification is None


def test_route_request_returns_chat_without_clarifying_for_an_empty_query():
    decision = ir.route_request("   ", llm=None, explicit_source=False)

    assert decision.strategy is RouteStrategy.CHAT
    assert decision.clarification is None


def test_route_query_answer_is_unchanged_at_every_guess_site():
    class _GarbageLLM:
        def complete(self, messages, **kwargs):
            return ""

    assert (
        route_query("Review the vendor renewal terms", llm=None, explicit_source=False)
        is RouteStrategy.CHAT
    )
    assert (
        route_query(
            "Review the vendor renewal terms",
            llm=_GarbageLLM(),
            explicit_source=False,
        )
        is RouteStrategy.CHAT
    )
    assert (
        route_query("email the quarterly report to legal", llm=None, explicit_source=False)
        is RouteStrategy.TOOL
    )


def test_route_request_honors_the_clarification_setting():
    settings = AppSettings(route_clarification=False)
    decision = ir.route_request(
        "Review the vendor renewal terms",
        llm=None,
        explicit_source=False,
        settings=settings,
    )

    assert decision.strategy is RouteStrategy.CHAT
    assert decision.clarification is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/servers/web/test_agent_router.py -q`

Expected: FAIL with `AttributeError: module 'src.internal.servers.web.intent_routing' has no attribute 'route_request'`.

- [ ] **Step 3: Add the typed configuration**

In `src/internal/configs/app_configs.py`, add the field to `AppSettings` beside `intent_model_min_confidence`:

```python
    route_clarification: bool = True
```

In `load_app_settings`, pass it through with `get_env_bool`, the helper already used in that function for `AGENTIC_SEARCH_DEV_ADMIN`, `DISABLE_VECTOR_DB`, and `MULTI_TENANT`:

```python
        route_clarification=get_env_bool(
            source, "AGENTIC_SEARCH_ROUTE_CLARIFICATION", True
        ),
```

In `src/internal/configs/default_config.py`, beside the intent-model keys:

```python
    "AGENTIC_SEARCH_ROUTE_CLARIFICATION": True,
```

- [ ] **Step 4: Add the clarification types**

In `src/internal/servers/web/intent_routing.py`, after the `RouteStrategy` class:

```python
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
    question=(
        "I can take this a few different ways — which would you like?"
    ),
    options=(
        ClarificationOption("chat", "Explain or summarize it"),
        ClarificationOption("search", "Find the document or facts"),
        ClarificationOption("tool", "Take an action on it"),
    ),
)
```

Add `from dataclasses import dataclass` to the imports.

- [ ] **Step 5: Split the heuristic so it reports a no-signal default**

Replace `_rule_based_route` with a pair, keeping the existing docstring on the wrapper:

```python
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
```

An empty query returns `CHAT` from `_rule_based_route_or_none`, not `None`, so it is treated as decided and never clarifies.

- [ ] **Step 6: Let the classifier report an unusable answer**

In `classify_route`, change the return type annotation to `"tuple[RouteStrategy | None, dict]"` and return `None` instead of the chat default. The final line becomes:

```python
    return strategy, {"raw_label": captured_label}
```

where `strategy` is initialized to `None` instead of `RouteStrategy.CHAT`:

```python
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
```

Update its docstring to say it returns `None` when the response is empty or unexpected, and that the caller supplies the `chat` default.

- [ ] **Step 7: Implement route_request and reduce route_query to a shim**

Replace the body of `route_query` with `route_request`, keeping the existing cascade order, `_record_intent` calls, capture stage, and telemetry writes exactly as they are. The two guess-sites now attach `_CLARIFICATION`:

```python
def route_request(
    query: str,
    *,
    llm: "LLMClient | None",
    explicit_source: bool,
    settings: "AppSettings | None" = None,
    telemetry: dict | None = None,
) -> RouteDecision:
    """Decide the agent strategy, or ask when the cascade has no signal."""

    def decided(mechanism: str, strategy: RouteStrategy, detail: dict) -> RouteDecision:
        _record_intent(mechanism, strategy, detail)
        if telemetry is not None:
            telemetry["route_mechanism"] = mechanism
        return RouteDecision(strategy)

    def guessed(strategy: RouteStrategy, detail: dict) -> RouteDecision:
        if settings is not None and not settings.route_clarification:
            return decided("heuristic_default", strategy, detail)
        _record_intent("clarify", strategy, detail)
        if telemetry is not None:
            telemetry["route_mechanism"] = "clarify"
        return RouteDecision(strategy, _CLARIFICATION)

    if explicit_source:
        return decided("explicit_source", RouteStrategy.SEARCH, {})
    regex_choice = _regex_route(query)
    if regex_choice is not None:
        return decided("rules", regex_choice, {})
    fallback_detail: dict = {}
    model_choice = predict_route(query, settings=settings)
    if model_choice is not None:
        abstained = model_choice.confidence < model_choice.threshold
        model_detail = {
            "predicted_intent": model_choice.strategy.value,
            "confidence": model_choice.confidence,
            "threshold": model_choice.threshold,
            "abstained": abstained,
            "fallback_reason": "model_below_threshold" if abstained else None,
            "latency_ms": model_choice.latency_ms,
        }
        _capture.record_stage("intent_model", "evaluation", model_detail)
        if telemetry is not None:
            telemetry.update(
                route_predicted_intent=model_choice.strategy.value,
                route_confidence=model_choice.confidence,
                route_threshold=model_choice.threshold,
                route_abstained=abstained,
                route_model_latency_ms=model_choice.latency_ms,
            )
        if not abstained:
            return decided("model", model_choice.strategy, model_detail)
        fallback_detail = {"fallback_reason": "model_below_threshold"}
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
```

Move `route_query`'s existing cascade docstring onto `route_request` and extend it with the two guess-sites.

- [ ] **Step 8: Run the router and config tests**

Run: `python -m pytest tests/unit/servers/web/test_agent_router.py tests/unit/servers/web/test_stage_emits_intent.py tests/unit/test_configs.py -q`

Expected: PASS. Existing `route_query` assertions must pass unmodified; if any needs editing, the shim is not equivalent — fix the shim, not the test.

- [ ] **Step 9: Commit**

```bash
git add src/internal/servers/web/intent_routing.py src/internal/configs/app_configs.py src/internal/configs/default_config.py tests/unit/servers/web/test_agent_router.py tests/unit/test_configs.py
git commit -m "feat(routing): let the cascade report when a route was a guess"
```

---

### Task 2: Rename the mechanism labels

**Files:**
- Modify: `src/internal/servers/web/intent_routing.py`
- Modify: `tests/unit/servers/web/test_agent_router.py`
- Modify: `tests/unit/servers/web/test_stage_emits_intent.py`
- Modify: `docs/request-routing.md`

**Interfaces:**
- Consumes: `route_request` from Task 1.
- Produces: mechanism label vocabulary `explicit_source`, `rules`, `model`, `classifier`, `heuristic_default`, `clarify`, `user_selected`.

Task 1 already emits `rules`, `heuristic_default`, and `clarify` in the new code. This task removes every remaining use of the old names and pins the vocabulary.

- [ ] **Step 1: Write the failing vocabulary test**

```python
def test_route_mechanisms_use_the_documented_vocabulary():
    telemetry: dict = {}
    route_request("find the onboarding checklist", llm=None, explicit_source=False, telemetry=telemetry)
    assert telemetry["route_mechanism"] == "rules"

    telemetry = {}
    route_request("email the quarterly report to legal", llm=None, explicit_source=False, telemetry=telemetry)
    assert telemetry["route_mechanism"] == "heuristic_default"

    telemetry = {}
    route_request("Review the vendor renewal terms", llm=None, explicit_source=False, telemetry=telemetry)
    assert telemetry["route_mechanism"] == "clarify"
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/unit/servers/web/test_agent_router.py::test_route_mechanisms_use_the_documented_vocabulary -q`

Expected: PASS if Task 1 is complete. If it fails, Task 1 left an old label in place — fix that.

- [ ] **Step 3: Remove every remaining old label**

Run: `rg -n '"regex"|"rule_based"|regex · |rule_based' src/ tests/ docs/`

Replace each hit with `rules` / `heuristic_default`. Function names such as `_regex_route` and `_rule_based_route` keep their current names — only the emitted strings change.

- [ ] **Step 4: Document the vocabulary**

In `docs/request-routing.md`, replace the mechanism list with a table:

```markdown
| Mechanism | Meaning |
|---|---|
| `explicit_source` | An explicit non-default source provider forced search |
| `rules` | Deterministic high-precision cues decided |
| `model` | The trained intent model was confident |
| `classifier` | The LLM classifier returned a usable label |
| `heuristic_default` | Nothing else worked; a heuristic cue decided |
| `clarify` | No signal at all; the user was asked |
| `user_selected` | The user chose the route |
```

- [ ] **Step 5: Run the affected tests**

Run: `python -m pytest tests/unit/servers/web/ -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/internal/servers/web/intent_routing.py tests/unit/servers/web docs/request-routing.md
git commit -m "refactor(routing): name mechanisms for what they mean"
```

---

### Task 3: Clarification through /api/agent

**Files:**
- Modify: `src/internal/servers/web/app.py`
- Modify: `tests/unit/servers/web/test_web_experience_app.py`
- Modify: `docs/request-routing.md`

**Interfaces:**
- Consumes: `route_request`, `RouteDecision`, `Clarification` from Task 1.
- Produces: `AgentExperienceRequest.route: str | None` accepting `"chat" | "search" | "tool"`.
- Produces: `AgentExperienceResponse.clarification: dict | None`.
- Produces: `_run_auto_routed(..., forced_route: RouteStrategy | None = None)`.
- Produces: SSE `done` event field `clarification`.

- [ ] **Step 1: Write the failing endpoint tests**

These tests only clarify when the app has no LLM client — a configured LLM
returns a usable label and decides the route. `create_web_app` can pick one up
two ways, both closed by the existing
`test_agent_no_llm_chat_degrades_to_pipeline` in this file: inject empty
`app_settings` so the config loader cannot set `resolved.llm.api_key`, and set
`OPENAI_API_KEY=""` (python-dotenv will not override an already-present var, so
`delenv` alone is insufficient — `create_web_app`'s internal `.env` reload
re-adds it). Follow that test's setup exactly in both clarification tests below.

```python
def test_uncertain_query_asks_instead_of_running_an_agent(monkeypatch, tmp_path):
    from src.internal.configs import AppSettings

    def unexpected(*args, **kwargs):
        raise AssertionError("a clarification must not run an agent")

    monkeypatch.setattr("src.internal.servers.web.app._run_agentic_rag", unexpected)
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_search_direct_or_escalate", unexpected
    )
    monkeypatch.setattr("src.internal.servers.web.app._run_tool_agent", unexpected)

    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "clarify.sqlite3"),
        app_settings=AppSettings(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/agent", json={"query": "Review the vendor renewal terms"}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "clarify"
    assert [option["route"] for option in body["clarification"]["options"]] == [
        "chat",
        "search",
        "tool",
    ]
    assert body["documents"] == []
    assert body["hook_metadata"]["route_mechanism"] == "clarify"


def test_selected_route_dispatches_to_the_matching_runner(monkeypatch, tmp_path):
    calls = []

    async def fake_search(query, **kwargs):
        calls.append(("search", query))
        return "search answer", [], [], "search", {}

    monkeypatch.setattr(
        "src.internal.servers.web.app._run_search_direct_or_escalate", fake_search
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "picked.sqlite3"))
    with TestClient(app) as client:
        response = client.post(
            "/api/agent",
            json={"query": "Review the vendor renewal terms", "route": "search"},
        )

    assert response.status_code == 200
    assert calls == [("search", "Review the vendor renewal terms")]
    assert response.json()["hook_metadata"]["route_mechanism"] == "user_selected"


def test_unknown_route_value_is_rejected(tmp_path):
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "bad.sqlite3"))
    with TestClient(app) as client:
        response = client.post(
            "/api/agent", json={"query": "anything", "route": "teleport"}
        )

    assert response.status_code == 422


def test_explicit_mode_never_clarifies(monkeypatch, tmp_path):
    called = {}

    async def fake_answer(*args, **kwargs):
        called["answer"] = True
        return "chat answer", [], [], {}

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "explicit.sqlite3"))
    with TestClient(app) as client:
        response = client.post(
            "/api/agent",
            json={"query": "Review the vendor renewal terms", "mode": "chat_once"},
        )

    assert response.status_code == 200
    assert called.get("answer") is True
    assert response.json()["intent"] == "chat"
    assert response.json()["clarification"] is None


def test_streaming_done_event_carries_the_clarification(tmp_path):
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "stream.sqlite3"))
    with TestClient(app) as client:
        response = client.post(
            "/api/agent/stream", json={"query": "Review the vendor renewal terms"}
        )

    assert response.status_code == 200
    done = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ][-1]
    assert done["type"] == "done"
    assert done["intent"] == "clarify"
    assert done["clarification"]["options"][1]["route"] == "search"
```

If `json` is not already imported in this test module, add `import json` at the top.

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/unit/servers/web/test_web_experience_app.py -q -k "clarif or selected_route or unknown_route"`

Expected: FAIL — the response has no `clarification` key and `route` is ignored.

- [ ] **Step 3: Add the request and response fields**

In `AgentExperienceRequest`, after `mode`:

```python
    route: str | None = Field(
        default=None,
        description=(
            "Optional route chosen by the user after a clarification: 'chat', "
            "'search', or 'tool'. Skips the router and dispatches directly."
        ),
    )
```

In `AgentExperienceResponse`, after `intent`:

```python
    clarification: dict | None = None
```

Add the normalizer beside `_normalize_agent_mode`:

```python
_VALID_ROUTES = {"chat", "search", "tool"}


def _normalize_route(route: str) -> RouteStrategy:
    requested = route.strip().lower()
    if requested not in _VALID_ROUTES:
        valid = ", ".join(sorted(_VALID_ROUTES))
        raise HTTPException(status_code=422, detail=f"route must be one of: {valid}")
    return RouteStrategy(requested)
```

- [ ] **Step 4: Carry the clarification through the response tail**

In the response-building helper, next to `tool_calls = extra.pop("tool_calls", [])`:

```python
    clarification = extra.pop("clarification", None)
```

and pass it to the constructed `AgentExperienceResponse`:

```python
        clarification=clarification,
```

This reuses the established pattern where a response field travels on `extra` and is popped before `extra` is merged into `hook_metadata`.

- [ ] **Step 5: Short-circuit the clarification in the dispatcher**

In `_run_auto_routed`, add the keyword-only parameter `forced_route: RouteStrategy | None = None` and replace the `route_query` call:

```python
    if forced_route is not None:
        strategy = forced_route
        extra["route_mechanism"] = "user_selected"
    else:
        decision = route_request(
            query,
            llm=llm,
            explicit_source=explicit_source,
            settings=app_settings,
            telemetry=extra,
        )
        if decision.clarification is not None:
            extra["route"] = "clarify"
            extra["clarification"] = {
                "question": decision.clarification.question,
                "options": [
                    {"route": option.route, "label": option.label}
                    for option in decision.clarification.options
                ],
            }
            return decision.clarification.question, [], [], "clarify", extra
        strategy = decision.strategy
    extra["route"] = strategy.value
```

The early return uses the same five-element tuple every other branch returns, so no caller changes.

- [ ] **Step 6: Pass the selected route from the endpoint**

At the `_run_auto_routed` call site, add:

```python
                        forced_route=(
                            _normalize_route(request.route) if request.route else None
                        ),
```

- [ ] **Step 7: Add the field to the streaming done event**

In the SSE `done` payload, after `"intent": result.intent,`:

```python
                        "clarification": result.clarification,
```

- [ ] **Step 8: Run the endpoint tests**

Run: `python -m pytest tests/unit/servers/web/test_web_experience_app.py -q`

Expected: PASS.

- [ ] **Step 9: Document the round trip**

In `docs/request-routing.md`, add after the cascade description:

```markdown
When no step in the cascade has a signal, the router asks instead of guessing.
The response carries `intent: "clarify"` and a `clarification` object holding a
question and one option per route; no agent runs. Sending the same query back
with `route` set to `chat`, `search`, or `tool` skips the router and dispatches
through the normal auto path, so the selected agent and its degradation
behavior are identical. Set `AGENTIC_SEARCH_ROUTE_CLARIFICATION=false` to
restore the previous behavior of always choosing a route.
```

- [ ] **Step 10: Commit**

```bash
git add src/internal/servers/web/app.py tests/unit/servers/web/test_web_experience_app.py docs/request-routing.md
git commit -m "feat(routing): ask the user when no route has a signal"
```

---

### Task 4: Clarification in the Assistant UI

**Files:**
- Modify: `web/src/types.ts`
- Modify: `web/src/pages/AssistPage.tsx`
- Create: `web/src/components/ClarificationPrompt.tsx`
- Create: `web/src/components/__tests__/ClarificationPrompt.test.tsx`

**Interfaces:**
- Consumes: the `clarification` field on the agent response and SSE `done` event from Task 3.
- Produces: `ClarificationPrompt({ clarification, onSelect })`.

- [ ] **Step 1: Write the failing component test**

```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ClarificationPrompt } from "../ClarificationPrompt";

describe("ClarificationPrompt", () => {
  const clarification = {
    question: "I can take this a few different ways — which would you like?",
    options: [
      { route: "chat", label: "Explain or summarize it" },
      { route: "search", label: "Find the document or facts" },
      { route: "tool", label: "Take an action on it" },
    ],
  };

  it("asks the question and reports the chosen route", async () => {
    const onSelect = vi.fn();
    render(
      <ClarificationPrompt clarification={clarification} onSelect={onSelect} />,
    );

    expect(screen.getByText(clarification.question)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Find the document or facts" }));

    expect(onSelect).toHaveBeenCalledWith("search");
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd web && npx vitest run src/components/__tests__/ClarificationPrompt.test.tsx`

Expected: FAIL — cannot resolve `../ClarificationPrompt`.

- [ ] **Step 3: Add the types**

In `web/src/types.ts`:

```ts
export interface ClarificationOptionView {
  route: "chat" | "search" | "tool";
  label: string;
}

export interface ClarificationView {
  question: string;
  options: ClarificationOptionView[];
}
```

Add `route?: "chat" | "search" | "tool";` to `AgentExperienceRequest`, and `clarification?: ClarificationView | null;` to the agent response interface and the SSE done-event interface. Widen the two `intent?: "search" | "chat" | "tool"` unions to include `"clarify"`.

- [ ] **Step 4: Write the component**

```tsx
import type { ClarificationView } from "../types";

export function ClarificationPrompt({
  clarification,
  onSelect,
}: {
  clarification: ClarificationView;
  onSelect: (route: "chat" | "search" | "tool") => void;
}) {
  return (
    <section className="clarification" aria-label="Clarify the request">
      <p className="clarification-question">{clarification.question}</p>
      <div className="clarification-options">
        {clarification.options.map((option) => (
          <button
            key={option.route}
            type="button"
            className="icon-button"
            onClick={() => onSelect(option.route)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </section>
  );
}
```

- [ ] **Step 5: Run the component test**

Run: `cd web && npx vitest run src/components/__tests__/ClarificationPrompt.test.tsx`

Expected: PASS.

- [ ] **Step 6: Wire it into AssistPage**

Add state beside the existing `intent` state at line 45, keeping the asked query with the question so the follow-up re-submits the right text:

```tsx
const [clarification, setClarification] = useState<
  { view: ClarificationView; query: string } | null
>(null);
```

In the SSE handler where `event.intent` is read (line 157), also record the clarification, and clear it at the start of a submission alongside the existing answer reset:

```tsx
setClarification(
  event.clarification ? { view: event.clarification, query: normalizedQuery } : null,
);
```

`handleSubmit` at line 88 already accepts either a `FormEvent` or a query string. Add a second optional parameter rather than a new request path:

```tsx
const handleSubmit = useCallback(
  async (eventOrQuery?: FormEvent | string, options?: { route?: "chat" | "search" | "tool" }) => {
```

and include it in the `agentRequest` object it builds at line 123:

```tsx
        route: options?.route,
```

Render the prompt above the answer column:

```tsx
{clarification && (
  <ClarificationPrompt
    clarification={clarification.view}
    onSelect={(route) => {
      const asked = clarification.query;
      setClarification(null);
      void handleSubmit(asked, { route });
    }}
  />
)}
```

- [ ] **Step 7: Type-check and run the frontend tests**

Run: `cd web && npm run typecheck && npx vitest run`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add web/src/types.ts web/src/pages/AssistPage.tsx web/src/components/ClarificationPrompt.tsx web/src/components/__tests__/ClarificationPrompt.test.tsx
git commit -m "feat(web): let the assistant ask which route the user meant"
```

---

### Task 5: Model-scoped tool precision gate

**Files:**
- Modify: `src/model/intent_evaluation.py`
- Modify: `tests/unit/test_intent_evaluation.py`
- Modify: `docs/training-and-evaluation.md`

**Interfaces:**
- Consumes: `IntentEvaluationReport`, `PromotionCriteria`, `compare_for_promotion`.
- Produces: `IntentEvaluationReport.model_tool_precision: float | None`, included in `to_dict()`.

- [ ] **Step 1: Write the failing dilution test**

```python
def test_tool_precision_gate_ignores_deterministic_routes():
    """Regex-decided tool routes must not dilute the model's own precision.

    Forty correct deterministic tool routes plus two false model tool routes
    give 0.9524 cascade precision, which clears the 0.95 limit while the model
    itself got every tool route wrong.
    """
    records = [
        IntentPredictionRecord(f"r{i}", "tool", "tool", 1.0, 0.1, "regex")
        for i in range(40)
    ] + [
        IntentPredictionRecord(f"m{i}", "chat", "tool", 0.99, 0.1, "model")
        for i in range(2)
    ]
    report = evaluate_intent_predictions(records, threshold=0.5)

    assert report.tool_precision == pytest.approx(0.9524, abs=1e-4)
    assert report.model_tool_precision == 0.0

    decision = compare_for_promotion(
        report,
        _report(
            macro_f1=0.0,
            tool_precision=1.0,
            fallback_rate=1.0,
            p50_latency_ms=400.0,
        ),
        PromotionCriteria(max_high_confidence_errors=2),
    )

    assert "tool_precision_minimum" in decision.failed_gates


def test_tool_precision_is_unmeasured_when_the_model_predicts_no_tool_route():
    records = [IntentPredictionRecord("m1", "chat", "chat", 0.99, 0.1, "model")]
    report = evaluate_intent_predictions(records, threshold=0.5)

    assert report.model_tool_precision is None

    decision = compare_for_promotion(
        report,
        _report(
            macro_f1=0.0,
            tool_precision=1.0,
            fallback_rate=1.0,
            p50_latency_ms=400.0,
        ),
        PromotionCriteria(),
    )

    assert "tool_precision_minimum" in decision.failed_gates
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/unit/test_intent_evaluation.py -q -k tool_precision`

Expected: FAIL with `AttributeError: 'IntentEvaluationReport' object has no attribute 'model_tool_precision'`.

- [ ] **Step 3: Compute the model-scoped precision**

Add the field to `IntentEvaluationReport` after `out_of_scope_abstention`:

```python
    model_tool_precision: float | None = None
```

Add it to `to_dict()`:

```python
            "model_tool_precision": self.model_tool_precision,
```

In `evaluate_intent_predictions`, after `covered` is built:

```python
    covered_tool_predictions = sum(record.predicted == "tool" for record in covered)
    model_tool_precision = (
        sum(
            record.predicted == "tool" and record.expected == "tool"
            for record in covered
        )
        / covered_tool_predictions
        if covered_tool_predictions
        else None
    )
```

and pass `model_tool_precision=model_tool_precision` to the returned report. `None` means unmeasured — no covered record predicted `tool` — rather than scikit-learn's `0.0` for an empty prediction set.

- [ ] **Step 4: Point the gate at it**

In `compare_for_promotion`, replace the `tool_precision_minimum` gate:

```python
        _gate(
            "tool_precision_minimum",
            # Unmeasured is not evidence of safety, as with out-of-scope.
            candidate.model_tool_precision is not None
            and candidate.model_tool_precision >= criteria.min_tool_precision,
            candidate.model_tool_precision,
            criteria.min_tool_precision,
        ),
```

- [ ] **Step 5: Update the shared report helper and its callers**

The `_report(...)` helper at the top of `tests/unit/test_intent_evaluation.py` builds an `IntentEvaluationReport` directly, so every candidate it produces now has `model_tool_precision=None` and fails the gate. Add the parameter and thread it through:

```python
def _report(
    *,
    macro_f1: float,
    tool_precision: float,
    fallback_rate: float | None,
    p50_latency_ms: float | None,
    high_confidence_errors: int = 0,
    model_tool_precision: float | None = None,
) -> IntentEvaluationReport:
```

and pass `model_tool_precision=model_tool_precision` to the constructor. Then, in each existing test whose candidate is expected to clear the tool gate, pass `model_tool_precision=tool_precision`. Specifically `test_promotion_fails_when_macro_f1_regresses`, `test_promotion_reports_each_gate_and_rejects_missing_baseline_measurements`, and `test_authoritative_route_gate_fails_when_candidate_changes_regex_record` assert on exact `failed_gates` sets that would otherwise gain `tool_precision_minimum`.

- [ ] **Step 6: Run the evaluation and training tests**

Run: `python -m pytest tests/unit/test_intent_evaluation.py tests/unit/test_intent_training.py -q`

Expected: PASS. A fixture asserting a promotable candidate needs a covered record predicting `tool`; add one rather than weakening the gate.

- [ ] **Step 7: Document the change**

In `docs/training-and-evaluation.md`, in the paragraph listing the promotion gates, replace the tool-precision clause with:

```markdown
the configured minimum `tool` precision measured over the model's own covered
predictions, so deterministic routes cannot dilute it — and reported as `null`,
which fails the gate, when the model predicted no `tool` route at all;
```

- [ ] **Step 8: Commit**

```bash
git add src/model/intent_evaluation.py tests/unit/test_intent_evaluation.py docs/training-and-evaluation.md
git commit -m "fix(intent): gate on the model's own tool precision"
```

---

### Task 6: Configuration docs and full verification

**Files:**
- Modify: `docs/configuration.md`
- Modify: `.env.example`

**Interfaces:**
- Documents `AGENTIC_SEARCH_ROUTE_CLARIFICATION` from Task 1.

- [ ] **Step 1: Document the setting**

In the routing table in `docs/configuration.md`:

```markdown
| `AGENTIC_SEARCH_ROUTE_CLARIFICATION` | Ask the user which route was meant when no step in the cascade has a signal; `true` by default. Set `false` to always choose a route, as before. |
```

Add the same key to the environment-variable reference table with default `true`, and to `.env.example`:

```bash
# Ask which route was meant when the router has no signal, instead of guessing.
AGENTIC_SEARCH_ROUTE_CLARIFICATION=true
```

- [ ] **Step 2: Run the lint gates**

```bash
python -m ruff check src/ tests/
python -m ruff format --check src/internal/servers/web/intent_routing.py src/internal/servers/web/app.py src/internal/configs/app_configs.py src/model/intent_evaluation.py
```

Expected: exit `0`.

- [ ] **Step 3: Confirm no old mechanism label survives**

```bash
rg -n '"regex"|"rule_based"' src/ tests/ docs/
```

Expected: no matches.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`

Expected: PASS, with no fewer tests than the 3044 passing before this plan. Record the full node ID and error for any environment-dependent failure separately; do not weaken a test to make it pass.

- [ ] **Step 5: Run the frontend gates**

Run: `cd web && npm run typecheck && npx vitest run`

Expected: PASS.

- [ ] **Step 6: Verify the four reference requests end to end**

```bash
python - <<'PY'
from src.internal.servers.web.intent_routing import route_request
for text in [
    "Explain how FAISS works",
    "Find the onboarding checklist",
    "Create a ticket for this problem",
    "Review the vendor renewal terms",
]:
    d = route_request(text, llm=None, explicit_source=False)
    print(f"{d.strategy.value:7} clarify={d.clarification is not None}  {text}")
PY
```

Expected: the first three route with `clarify=False`; the fourth reports `clarify=True`.

- [ ] **Step 7: Commit**

```bash
git add docs/configuration.md .env.example
git commit -m "docs: document the route clarification setting"
```

- [ ] **Step 8: Final review checkpoint**

Review the full branch diff against every success criterion in `docs/superpowers/specs/2026-08-12-route-clarification-design.md`. Confirm that no request reaching a signal changed its route, that `RouteStrategy` still has three members, and that no chat, search, tool, or MCP execution changed beyond the clarification short-circuit and the forced-route entry.
