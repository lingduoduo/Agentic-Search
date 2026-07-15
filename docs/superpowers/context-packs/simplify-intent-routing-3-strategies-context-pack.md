# Generated Context Pack

# Simplify Intent Routing 3 Strategies

## Sources

- [Specification: 2026-07-01-simplify-intent-routing-3-strategies-design.md](../specs/2026-07-01-simplify-intent-routing-3-strategies-design.md)
- [Plan: 2026-07-01-simplify-intent-routing-3-strategies.md](../plans/2026-07-01-simplify-intent-routing-3-strategies.md)

## Specification Context

### Goal

Collapse the entry-point router's **4** `RouteStrategy` values to the **3**
user-facing intents the README advertises (`search` / `chat` / `tool`), by
merging `direct_llm` and `agentic_rag` into a single grounded `chat` strategy.
This removes a two-vocabulary mismatch (4 internal strategies vs 3 surfaced
intents), deletes an ungrounded parametric path, and drops one orphaned helper.

### Out of scope (unchanged)

- `_infer_intent_from_output` — still used by the explicit tool-agent finalize
  path (`app.py:619`); keeps mapping the first tool call to a surfaced intent.
- The LLM classifier `classify_route` — kept; only its label set shrinks (drops
  `direct_llm`). The user chose to keep classification, not go pure-rule-based.
- Capability-aware degradation in `_run_auto_routed`.
- `IntentPipeline` (`src/model/intent_classifier.py`) — a separate ML model that
  tunes retrieval settings; unrelated to the entry-point router.

### Tests

- `test_intent_routing.py`: drop the `_rule_based_is_search` tests; update
  route-classification tests so former `direct_llm` cases (`write`, `translate`,
  greetings) now assert `RouteStrategy.CHAT`.
- `test_sse_streaming.py`: the `route == "direct_llm"` assertion becomes the new
  vocabulary (`chat` for that query, or an updated fixture query).
- New: a generative query (e.g. "write a haiku") routes to `CHAT` and
  `_run_auto_routed` returns a non-crashing answer even when retrieval yields
  zero relevant documents (guards the accepted-tradeoff behavior).

## Implementation Plan Context

### Global Constraints

- Do **not** touch the explicit registry mode names `search_agent` / `tool_agent` / `plain_generation` — those are agent-loop aliases in `src/agents/base.py`, unrelated to the `RouteStrategy` enum values being renamed.
- Do **not** modify `_infer_intent_from_output` — it maps the first tool call to a surfaced intent for the explicit tool-agent finalize path and is out of scope.
- Keep `classify_route` (LLM classifier) and the capability-aware degradation logic — only the label vocabulary changes.
- `route` (chosen strategy) and `intent` (what actually ran after degradation) share a vocabulary but remain distinct fields; they can legitimately differ.
- Every changed line must trace to this refactor. Remove only imports/helpers that *this* change orphans.
- Reference spec: `docs/superpowers/specs/2026-07-01-simplify-intent-routing-3-strategies-design.md`.

---

### Task 1: Delete the dead `_rule_based_is_search` helper

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

_[Section compacted.]_

### Task 2: Collapse `RouteStrategy` to `CHAT`/`SEARCH`/`TOOL` and remove the `direct_llm` path

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

_[Section compacted.]_

### Task 3: Guard the accepted tradeoff — a generative query routes to `chat` and dispatches cleanly

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

_[Section compacted.]_

### Task 4: Update the README

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

_[Section compacted.]_

### Final Verification

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

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
