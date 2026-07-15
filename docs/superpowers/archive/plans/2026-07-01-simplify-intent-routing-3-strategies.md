# Simplify Intent Routing: 4 Strategies → 3 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the entry-point router's four `RouteStrategy` values to the three user-facing intents (`chat`/`search`/`tool`) by merging `direct_llm` + `agentic_rag` into one grounded `chat` strategy, and delete the orphaned `_rule_based_is_search` helper.

**Architecture:** `RouteStrategy` is renamed to `CHAT`/`SEARCH`/`TOOL` with values matching the surfaced `intent`. The `DIRECT_LLM` classification label, its `_DIRECT_RE` regex, and its dispatch tail in `_run_auto_routed` are removed; a no-LLM `chat` query now degrades to `_auto_search_pipeline` exactly as the former `AGENTIC_RAG` no-LLM path did. The LLM classifier and capability-aware degradation are kept; only the label set shrinks.

**Tech Stack:** Python 3, pytest, FastAPI (web app), regex-based + LLM-backed classifier.

## Global Constraints

- Do **not** touch the explicit registry mode names `search_agent` / `tool_agent` / `plain_generation` — those are agent-loop aliases in `src/agents/base.py`, unrelated to the `RouteStrategy` enum values being renamed.
- Do **not** modify `_infer_intent_from_output` — it maps the first tool call to a surfaced intent for the explicit tool-agent finalize path and is out of scope.
- Keep `classify_route` (LLM classifier) and the capability-aware degradation logic — only the label vocabulary changes.
- `route` (chosen strategy) and `intent` (what actually ran after degradation) share a vocabulary but remain distinct fields; they can legitimately differ.
- Every changed line must trace to this refactor. Remove only imports/helpers that *this* change orphans.
- Reference spec: `docs/superpowers/specs/2026-07-01-simplify-intent-routing-3-strategies-design.md`.

---

## File Structure

- `src/internal/servers/web/intent_routing.py` — the enum, regexes, rule-based route, LLM classifier, `route_query`, and the dead `_rule_based_is_search` helper. Primary edit surface.
- `src/internal/servers/web/app.py` — `_run_auto_routed` dispatch (branch names + the deleted `DIRECT_LLM` tail).
- `tests/unit/test_intent_routing.py` — drop `_rule_based_is_search` tests.
- `tests/unit/servers/web/test_agent_router.py` — rename enum refs, drop `direct_llm` label.
- `tests/unit/test_execution_fallbacks.py`, `tests/unit/servers/web/test_tool_trace.py`, `tests/unit/servers/web/test_web_experience_app.py`, `tests/unit/servers/web/test_sse_streaming.py` — update `RouteStrategy.*` references and the removed-branch tests.
- `README.md` — Intent Routing table + auto-router strategy vocabulary.

---

## Task 1: Delete the dead `_rule_based_is_search` helper

Independent, self-contained: the helper is referenced only by tests. Do this first so the atomic rename in Task 2 has a smaller surface.

**Files:**
- Modify: `src/internal/servers/web/intent_routing.py:31-46`
- Modify: `tests/unit/test_intent_routing.py:8-11,28-59`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (pure deletion).

- [ ] **Step 1: Confirm the helper is production-unused**

Run: `grep -rn "_rule_based_is_search" src/`
Expected: exactly one hit — its `def` at `src/internal/servers/web/intent_routing.py:31`. (No production callers.)

- [ ] **Step 2: Delete the 7 helper tests and fix the import**

In `tests/unit/test_intent_routing.py`, change the import block (lines 8-11) from:

```python
from src.internal.servers.web.intent_routing import (
    _infer_intent_from_output,
    _rule_based_is_search,
)
```

to:

```python
from src.internal.servers.web.intent_routing import _infer_intent_from_output
```

Then delete the entire `# --- _rule_based_is_search ---` section (lines 28-59): the seven functions `test_rule_based_is_search_find_keyword`, `test_rule_based_is_search_short_keyword_query`, `test_rule_based_is_search_list_keyword`, `test_rule_based_is_chat_explain_keyword`, `test_rule_based_is_chat_what_is`, `test_rule_based_is_chat_default_no_signal`, `test_rule_based_is_search_show_me`.

- [ ] **Step 3: Delete the helper**

In `src/internal/servers/web/intent_routing.py`, delete the `_rule_based_is_search` function (lines 31-46), the whole block:

```python
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
```

Leave `_SEARCH_RE`, `_CHAT_RE`, `_VERB_RE` in place — they are still used by `_is_bare_lookup`.

- [ ] **Step 4: Verify no dangling references**

Run: `grep -rn "_rule_based_is_search" src/ tests/`
Expected: no output.

- [ ] **Step 5: Run the affected test file**

Run: `pytest tests/unit/test_intent_routing.py -q`
Expected: PASS (remaining `_infer_intent_from_output` and routing-tool tests green).

- [ ] **Step 6: Commit**

```bash
git add src/internal/servers/web/intent_routing.py tests/unit/test_intent_routing.py
git commit -m "refactor(routing): remove dead _rule_based_is_search helper"
```

---

## Task 2: Collapse `RouteStrategy` to `CHAT`/`SEARCH`/`TOOL` and remove the `direct_llm` path

This is one atomic rename: the enum is imported by `app.py` and five test files, so the source change and all consumer updates land together to keep the suite green. Update tests first (TDD), watch them fail against the old code, then change the source.

**Files:**
- Modify: `src/internal/servers/web/intent_routing.py` (enum, `_DIRECT_RE`, `_is_bare_lookup`, `_rule_based_route`, `_ROUTE_PROMPT`, `classify_route`, `route_query`)
- Modify: `src/internal/servers/web/app.py:678-817` (`_run_auto_routed`)
- Modify: `tests/unit/servers/web/test_agent_router.py`
- Modify: `tests/unit/test_execution_fallbacks.py`
- Modify: `tests/unit/servers/web/test_tool_trace.py`
- Modify: `tests/unit/servers/web/test_web_experience_app.py`
- Modify: `tests/unit/servers/web/test_sse_streaming.py`

**Interfaces:**
- Produces: `RouteStrategy` with exactly three members — `CHAT = "chat"`, `SEARCH = "search"`, `TOOL = "tool"`. `route_query(query, *, llm, has_local_model, explicit_source) -> RouteStrategy` (signature unchanged). `_rule_based_route(query) -> RouteStrategy`. `classify_route(query, llm) -> RouteStrategy`.
- Consumes: `_run_auto_routed` dispatches on the three members; `extra["route"] == strategy.value ∈ {"chat","search","tool"}`.

- [ ] **Step 1: Update `test_agent_router.py` to the new vocabulary**

Rewrite every `RouteStrategy.*` reference and the `classify_route` label expectations. Replace the file's route-assertion bodies as follows.

The cascade tests (`SEARCH_AGENT` → `SEARCH`):

```python
    assert strategy is RouteStrategy.SEARCH
```

for `test_route_query_explicit_source_is_search`, `test_route_query_without_llm_uses_rule_based`, `test_route_query_bare_lookup_is_search_and_skips_classifier`.

`test_route_query_uses_llm_classifier_when_available` — the fake LLM returns a label; change it to `"tool"` and assert:

```python
    assert strategy is RouteStrategy.TOOL
```

`test_route_query_descriptive_phrase_still_uses_classifier` — fake LLM returns `"chat"`; assert:

```python
    assert strategy is RouteStrategy.CHAT
```

`_rule_based_route` tests:

```python
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
```

`classify_route` tests:

```python
def test_classify_route_parses_each_label():
    for label, expected in [
        ("chat", RouteStrategy.CHAT),
        ("search", RouteStrategy.SEARCH),
        ("tool", RouteStrategy.TOOL),
    ]:
        assert classify_route("q", _FakeLLM(label)) is expected


def test_classify_route_defaults_to_chat_on_garbage():
    assert classify_route("q", _FakeLLM("nonsense reply")) is RouteStrategy.CHAT
    assert classify_route("q", _FakeLLM("")) is RouteStrategy.CHAT
```

- [ ] **Step 2: Run `test_agent_router.py` to verify it fails against old code**

Run: `pytest tests/unit/servers/web/test_agent_router.py -q`
Expected: FAIL — `AttributeError: SEARCH` / `CHAT` / `TOOL` (the enum still has the old members).

- [ ] **Step 3: Rewrite the enum, regexes, rule route, prompt, and classifier**

In `src/internal/servers/web/intent_routing.py`:

Replace the `RouteStrategy` class (lines 78-84):

```python
class RouteStrategy(str, Enum):
    """High-level agent strategy chosen by the entry-point router.

    Values match the user-facing ``intent`` vocabulary so ``extra["route"]``
    (the chosen strategy) reads the same as the surfaced ``intent`` (what
    actually ran after degradation).
    """

    CHAT = "chat"  # grounded synthesis via AgenticRAGLoop / degraded pipeline
    SEARCH = "search"  # multi-turn search until evidence suffices
    TOOL = "tool"  # OpenAPI / MCP function calling
```

Delete the `_DIRECT_RE` regex (lines 94-99):

```python
# Conversational / generative asks that need no retrieval.
_DIRECT_RE = re.compile(
    r"\b(write|translate|rephrase|reword|rewrite|draft|brainstorm|"
    r"hello|hi there|thanks|joke|poem|haiku)\b",
    re.IGNORECASE,
)
```

In `_is_bare_lookup`, remove the `_DIRECT_RE` clause from the exclusion guard (lines 114-121) so it reads:

```python
    if (
        _TOOL_RE.search(q)
        or _SEARCH_RE.search(q)
        or _CHAT_RE.search(q)
        or _VERB_RE.search(q)
    ):
        return False
    return len(q.split()) <= 3
```

Replace `_rule_based_route` (lines 125-144):

```python
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
```

Replace `_ROUTE_PROMPT` (lines 147-163):

```python
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
```

In `classify_route`, change both `agentic_rag` fallbacks to `CHAT` (lines 181-192):

```python
    if not content:
        logger.warning("Route classification empty; defaulting to chat.")
        return RouteStrategy.CHAT
    for value, strategy in _LABEL_BY_VALUE.items():
        if value in content:
            return strategy
    logger.warning(
        "Route classification returned unexpected response %r; defaulting to "
        "chat.",
        content,
    )
    return RouteStrategy.CHAT
```

In `route_query`, update the two deterministic returns and the docstring note (lines 217-227): `explicit_source` and `_is_bare_lookup` return `RouteStrategy.SEARCH`; the error fallback stays `_rule_based_route`. Update the docstring line that mentions "over-route such lookups to direct_llm" to "to chat/direct answers (ungrounded)".

- [ ] **Step 4: Run `test_agent_router.py` to verify it passes**

Run: `pytest tests/unit/servers/web/test_agent_router.py -q`
Expected: PASS.

- [ ] **Step 5: Update the dispatch branch names in `_run_auto_routed`**

In `src/internal/servers/web/app.py`:

Line 714: `if strategy is RouteStrategy.TOOL_AGENT:` → `if strategy is RouteStrategy.TOOL:`
Line 739 (tool degrade target): `strategy = RouteStrategy.AGENTIC_RAG` → `strategy = RouteStrategy.CHAT`
Line 742: `if strategy is RouteStrategy.SEARCH_AGENT:` → `if strategy is RouteStrategy.SEARCH:`
Line 770: `if strategy is RouteStrategy.AGENTIC_RAG:` → `if strategy is RouteStrategy.CHAT:`
Also update the branch comment on line 769 to `# ---- CHAT: grounded synthesis via AgenticRAGLoop (degrade to pipeline) ----`.

- [ ] **Step 6: Delete the `DIRECT_LLM` dispatch tail**

Delete lines 795-817 in `src/internal/servers/web/app.py` — the entire block from the `# ---- DIRECT_LLM: parametric answer, no retrieval ----` comment through the terminal `raise HTTPException(...)`:

```python
    # ---- DIRECT_LLM: parametric answer, no retrieval ----
    if llm is not None:
        messages = [ChatMessage(role=m.role, content=m.content) for m in history] + [
            ChatMessage(role="user", content=query)
        ]
        response = await asyncio.to_thread(llm.complete, messages)
        answer = response if isinstance(response, str) else response.content
        return answer, [], [], "chat", extra
    if has_local_model:
        from src import get_registered_agent_loop, resolve_agent_name

        loop_cls = get_registered_agent_loop(resolve_agent_name("plain_generation"))
        loop = loop_cls(tokenizer=tokenizer, server_manager=manager)
        output = await loop.run(
            [{"role": "user", "content": query}],
            sampling_params={"temperature": 0.0, "max_tokens": 512},
            on_turn=on_turn,
        )
        return output.final_answer or "", [], [], "chat", extra
    raise HTTPException(
        status_code=400,
        detail="No LLM configured. Set OPENAI_API_KEY or equivalent in .env.",
    )
```

The `CHAT` block (formerly `AGENTIC_RAG`, lines 770-793) returns in both the `llm is not None` and the no-LLM (`_auto_search_pipeline`) cases, and `TOOL` degrades into it — so the function always returns without the tail.

- [ ] **Step 7: Remove any import orphaned by the deletion**

Run: `grep -n "ChatMessage" src/internal/servers/web/app.py`
If the only remaining hit is the import line, remove `ChatMessage` from that import. If `ChatMessage` still appears elsewhere in `app.py`, leave the import.
Run: `grep -n "HTTPException\|asyncio\." src/internal/servers/web/app.py`
Expected: multiple other hits — leave both imports in place (they are used by other endpoints).

- [ ] **Step 8: Update `test_tool_trace.py` (`TOOL_AGENT` → `TOOL`)**

In `tests/unit/servers/web/test_tool_trace.py`, replace all five `lambda *a, **k: RouteStrategy.TOOL_AGENT` occurrences (lines 60, 82, 105, 125, 144) with `lambda *a, **k: RouteStrategy.TOOL`.

Run: `pytest tests/unit/servers/web/test_tool_trace.py -q`
Expected: PASS.

- [ ] **Step 9: Update `test_execution_fallbacks.py`**

In `tests/unit/test_execution_fallbacks.py`, replace every `RouteStrategy.SEARCH_AGENT` with `RouteStrategy.SEARCH`, every `RouteStrategy.TOOL_AGENT` with `RouteStrategy.TOOL`, and every `RouteStrategy.AGENTIC_RAG` with `RouteStrategy.CHAT` (in `_force_route` calls and the module/docstring prose at lines 3-6, 41, 45, 178, 182, 215, 257).

Run: `pytest tests/unit/test_execution_fallbacks.py -q`
Expected: PASS.

- [ ] **Step 10: Update the passing-route tests in `test_web_experience_app.py`**

In `tests/unit/servers/web/test_web_experience_app.py`:
- Line 599: `RouteStrategy.AGENTIC_RAG` → `RouteStrategy.CHAT` (and the docstring "AGENTIC_RAG route" → "CHAT route").
- Line 648: `RouteStrategy.SEARCH_AGENT` → `RouteStrategy.SEARCH`.
- Line 690: `RouteStrategy.TOOL_AGENT` → `RouteStrategy.TOOL`.

- [ ] **Step 11: Repurpose the removed-branch 400 test in `test_web_experience_app.py`**

The test at lines ~720-748 (`test_agent_no_llm...400`) exercised the deleted `DIRECT_LLM` tail. The new contract: a no-LLM `chat` query degrades to `_auto_search_pipeline` (route_degraded `"no_llm"`), never a "No LLM configured" 400. Replace the test body so it forces `CHAT`, provides no LLM, mocks `_auto_search_pipeline`, and asserts the degradation:

```python
def test_agent_no_llm_chat_degrades_to_pipeline(monkeypatch, tmp_path):
    """No LLM + CHAT route → _auto_search_pipeline (grounded degradation), not a 400."""
    from src.internal.servers.web.intent_routing import RouteStrategy

    monkeypatch.setattr(
        "src.internal.servers.web.app.route_query",
        lambda *a, **k: RouteStrategy.CHAT,
    )

    async def fake_pipeline(query, **kw):
        extra = kw.get("extra", {})
        return "extractive answer", ["[D1]"], [], "chat", extra

    monkeypatch.setattr(
        "src.internal.servers.web.app._auto_search_pipeline", fake_pipeline
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "explain FAISS"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "chat"
    assert body["answer"] == "extractive answer"
```

Confirm `_auto_search_pipeline`'s real signature before finalizing the mock: run `grep -n "async def _auto_search_pipeline" src/internal/servers/web/app.py` and match its keyword parameters (it is called with `query, llm=..., search_url=..., browser_search_url=..., rerank_url=..., top_k=..., filters=..., history=..., source_provider=..., extra=extra`). The `**kw` signature above absorbs them; keep it.

Run: `pytest tests/unit/servers/web/test_web_experience_app.py -q`
Expected: PASS (the model-loading path is not exercised because `route_query` and the pipeline are mocked).

- [ ] **Step 12: Update `test_sse_streaming.py` done-event route test**

In `tests/unit/servers/web/test_sse_streaming.py`, the test at lines 82-105 forced `DIRECT_LLM`. Rewrite it to force `CHAT` and mock `_run_agentic_rag` so the streamed done event carries the new vocabulary without running the real RAG loop:

```python
def test_stream_done_event_includes_route(monkeypatch, tmp_path):
    """The auto-route done event carries the chosen route + degradation."""
    from src.internal.servers.web.intent_routing import RouteStrategy

    monkeypatch.setattr(
        "src.internal.servers.web.app.route_query",
        lambda *a, **k: RouteStrategy.CHAT,
    )

    async def fake_rag(query, **kw):
        return "grounded answer", ["[D1]"], [], "chat", {}

    monkeypatch.setattr(
        "src.internal.servers.web.app._run_agentic_rag", fake_rag
    )

    class _LLM:
        def complete(self, messages, **_):
            return "unused"

    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"), llm=_LLM()
    )
    client = TestClient(app)

    resp = client.post("/api/agent/stream", json={"query": "explain FAISS"})
    assert resp.status_code == 200
    done_event = next(e for e in _parse_sse(resp.text) if e["type"] == "done")
    assert done_event["route"] == "chat"
    assert done_event["route_degraded"] is None
    assert done_event["intent"] == "chat"
```

Confirm the real `_run_agentic_rag` signature first: `grep -n "async def _run_agentic_rag" src/internal/servers/web/app.py` — it is called as `_run_agentic_rag(query, llm=llm, search_url=..., top_k=..., history=...)`; the `**kw` mock absorbs these.

Run: `pytest tests/unit/servers/web/test_sse_streaming.py -q`
Expected: PASS.

- [ ] **Step 13: Grep for any leftover old vocabulary**

Run: `grep -rn "DIRECT_LLM\|direct_llm\|AGENTIC_RAG\|SEARCH_AGENT\b\|TOOL_AGENT\b\|_DIRECT_RE" src/internal/servers/web/`
Expected: no output.
Run: `grep -rn "RouteStrategy\.\(DIRECT_LLM\|AGENTIC_RAG\|SEARCH_AGENT\|TOOL_AGENT\)" tests/`
Expected: no output.

- [ ] **Step 14: Run the full web + routing test scope**

Run: `pytest tests/unit/servers/web/ tests/unit/test_intent_routing.py tests/unit/test_execution_fallbacks.py -q`
Expected: PASS.

- [ ] **Step 15: Commit**

```bash
git add src/internal/servers/web/intent_routing.py src/internal/servers/web/app.py \
  tests/unit/servers/web/test_agent_router.py tests/unit/test_execution_fallbacks.py \
  tests/unit/servers/web/test_tool_trace.py tests/unit/servers/web/test_web_experience_app.py \
  tests/unit/servers/web/test_sse_streaming.py
git commit -m "refactor(routing): collapse RouteStrategy to chat/search/tool

Merge direct_llm + agentic_rag into one grounded CHAT strategy; drop the
DIRECT_LLM dispatch tail (no-LLM chat now degrades to the search pipeline).
route values now match the surfaced intent vocabulary."
```

---

## Task 3: Guard the accepted tradeoff — a generative query routes to `chat` and dispatches cleanly

Verifies the intended behavior: a generative ask ("write a haiku"), which formerly went to `direct_llm`, now routes to `CHAT` and is dispatched to the grounded path without error.

**Files:**
- Modify: `tests/unit/servers/web/test_web_experience_app.py` (add one test)

**Interfaces:**
- Consumes: `RouteStrategy.CHAT`, `_run_agentic_rag` (mocked), the `/api/agent` endpoint.

- [ ] **Step 1: Write the test**

Add to `tests/unit/servers/web/test_web_experience_app.py`:

```python
def test_generative_query_routes_to_chat_and_dispatches(monkeypatch, tmp_path):
    """A generative ask (former direct_llm) now routes to CHAT → grounded path,
    and dispatches cleanly even when retrieval yields zero documents."""
    dispatched = {}

    async def fake_rag(query, **kw):
        dispatched["query"] = query
        return "here is a haiku", [], [], "chat", {}

    monkeypatch.setattr("src.internal.servers.web.app._run_agentic_rag", fake_rag)

    class _LLM:
        def complete(self, messages, **_):
            return "chat"  # LLM classifier picks the chat label

    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"), llm=_LLM()
    )
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "write a haiku about the sea"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "chat"
    assert body["documents"] == []  # zero relevant docs, no crash
    assert dispatched["query"] == "write a haiku about the sea"
```

- [ ] **Step 2: Run it**

Run: `pytest tests/unit/servers/web/test_web_experience_app.py::test_generative_query_routes_to_chat_and_dispatches -q`
Expected: PASS. (If it fails because the `_LLM.complete` reply is not consulted — i.e. `route_query` short-circuits — verify the query is not caught by `_is_bare_lookup`; "write a haiku about the sea" is >3 tokens and matches no bare-lookup rule, so it reaches the classifier.)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/servers/web/test_web_experience_app.py
git commit -m "test(routing): generative query routes to chat and dispatches cleanly"
```

---

## Task 4: Update the README

Bring the Intent Routing docs in line with the 3-strategy router.

**Files:**
- Modify: `README.md:327-333` (Intent Routing table + description)
- Modify: `README.md:930-962` (auto-router strategy vocabulary)

**Interfaces:** none (docs).

- [ ] **Step 1: Fix the Intent Routing table and description (lines 327-333)**

Replace the table rows and the paragraph so `chat` maps to `AgenticRAGLoop` and the classifier is named correctly:

```markdown
| Intent | Agent loop | Trigger |
|--------|-----------|---------|
| `search` | `SearchAgentLoop` | Query needs external retrieval (web or indexed docs), or a bare entity lookup (e.g. `FAISS`) |
| `chat` | `AgenticRAGLoop` | Descriptive/conversational questions and generative asks — grounded synthesis |
| `tool` | `ToolAgentLoop` | Explicit tool use (`search_routing_tool`, custom tools) |

The router is `route_query` (`src/internal/servers/web/intent_routing.py`), dispatched by `_run_auto_routed` in `src/internal/servers/web/app.py`. It runs an LLM-backed 3-way classifier (`classify_route`) and falls back to a rule-based route (default `chat`) on ambiguous input.
```

- [ ] **Step 2: Fix the auto-router strategy vocabulary (lines 930-962)**

In the **Auto-router** paragraph, change the chosen-strategy list from `(direct_llm / agentic_rag / search_agent / tool_agent)` to `(chat / search / tool)`, and update the degradation examples: `search`→hybrid pipeline with no local model; `chat`→pipeline with no LLM. In the "Web reachability" list, replace `search_agent` with `search`, `agentic_rag` with `chat`, and delete the final bullet `- \`direct_llm\` performs no retrieval at all.` (the strategy no longer exists — generative asks now flow through grounded `chat`).

- [ ] **Step 3: Verify no stale strategy strings remain in the routing prose**

Run: `grep -n "direct_llm\|agentic_rag" README.md`
Expected: no hits in the auto-router / Intent Routing sections (mode aliases `search_agent`/`tool_agent` and file paths remain and are correct — do not touch those).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): align Intent Routing with 3-strategy router"
```

---

## Final Verification

- [ ] **Full suite**

Run: `pytest -q`
Expected: PASS (no regressions).

- [ ] **Lint**

Run: `ruff check . --fix && ruff format .`
Expected: clean.

- [ ] **Grep sweep for the removed vocabulary across the repo**

Run: `grep -rn "RouteStrategy\.\(DIRECT_LLM\|AGENTIC_RAG\|SEARCH_AGENT\|TOOL_AGENT\)\|_rule_based_is_search\|_DIRECT_RE" src/ tests/`
Expected: no output.

- [ ] **Success criteria (from the spec)**
  - `RouteStrategy` has exactly three members: `CHAT`, `SEARCH`, `TOOL`.
  - No `direct_llm` / `_rule_based_is_search` references remain in `src/`.
  - Router + web tests green, including the new generative-query test.
