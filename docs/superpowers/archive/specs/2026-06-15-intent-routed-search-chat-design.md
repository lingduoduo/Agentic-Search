# Intent-Routed Search & Chat — Design Spec

**Date:** 2026-06-15
**Status:** Approved

---

## Problem

The current `/api/agent` endpoint exposes six modes (`search_tool`, `hybrid_search`, `chat_once`, `chat_loop`, `search_agent`, `tool_agent`) behind a flat dropdown labeled "Entry Point". Users must understand the difference between them before submitting a query. Two modes (`search_agent`, `tool_agent`) silently require a local HuggingFace/vLLM model; without it they return a generic **"Agent search failed"** error. The `tool_agent` mode has a URL retrieval path that can error without a useful message.

---

## Goal

- Single input box — no mode selector visible to users.
- The system routes automatically: the `ToolAgentLoop` (local model) acts as the universal router when available, expressing intent by which tool it calls.
- Clean fallback when no local model is configured.
- Errors are specific and actionable, never "Agent search failed".

---

## Architecture

### Core Principle: The Tool Agent IS the Router

When a local model is available, every request flows through `ToolAgentLoop`. Intent is not classified separately — it emerges from which tool the model chooses to call:

```
User query
    ↓
ToolAgentLoop (local model)
    ├── calls search_tool()  → retrieval documents       → renders SourceGrid
    ├── calls rag_tool()     → synthesized RAG answer    → renders AnswerPanel
    └── calls any MCP tool   → arbitrary tool result     → renders trajectory + output
```

`search_tool` and `rag_tool` are registered as `FunctionTool` instances alongside existing MCP tools from `tool_registry`. The local model's tool-call decision is the intent.

### Fallback Chain (no local model)

When `search_agent_manager` is `None`, routing falls through three tiers in order. Each tier is only used if the previous one is unavailable or fails.

```
Tier 1 — ToolAgentLoop (local model)          [requires SEARCH_AGENT_MODEL]
    ↓ unavailable
Tier 2 — classify_is_search_flow (LLM)        [requires OPENAI_API_KEY or equivalent]
    ↓ unavailable or LLM call raises
Tier 3 — Rule-based keyword classifier        [always available, ~0ms]
    ↓ all fail
400 — "No LLM or local model configured"
```

**Tier 2 — LLM-backed binary classification**

Uses the existing `classify_is_search_flow(query, llm)`:

```
classify_is_search_flow → "search" → hybrid_search agent (intent = "search")
                        → "chat"   → answer_with_retrieval (intent = "chat")
```

If the LLM call raises any exception (timeout, API error, etc.), falls through to Tier 3 rather than propagating the error.

**Tier 3 — Rule-based classifier**

Keyword/pattern scoring, no external dependencies. Runs synchronously in <1ms.

| Signal | Intent |
|---|---|
| Starts with or contains: `find`, `list`, `retrieve`, `search for`, `show me`, `pull`, `get me` | `"search"` |
| Query is all keywords, no verb, ≤5 tokens | `"search"` |
| Contains: `explain`, `summarize`, `help me`, `write`, `what is`, `how do`, `why`, `difference between` | `"chat"` |
| No strong signal | `"chat"` (safe default — synthesized answer is more useful than empty search) |

`tool` mode is unavailable without a local model regardless of which tier runs. Explicit `mode="tool_agent"` in the request returns a 400 with a setup instruction.

**Routing pseudocode:**

```python
if local_model_available:
    # Tier 1
    output = await ToolAgentLoop(..., tools=[search_tool, rag_tool] + mcp_tools).run(...)
    intent = _infer_intent_from_trajectory(output.trajectory_messages)
elif llm is not None:
    # Tier 2, with Tier 3 as fallback
    try:
        is_search = classify_is_search_flow(query, llm)
    except Exception:
        logger.warning("LLM classifier failed, falling back to rule-based")
        is_search = _rule_based_is_search(query)
    if is_search:
        result = await _run_hybrid_search(...)
        intent = "search"
    else:
        result = await answer_with_retrieval(...)
        intent = "chat"
else:
    # Tier 3 only
    if _rule_based_is_search(query):
        result = await _run_hybrid_search(...)   # will 502 if retrieval server is down
        intent = "search"
    else:
        raise HTTPException(400, "No LLM configured. Set OPENAI_API_KEY in .env.")
```

The Tier 3-only path for `chat` raises immediately because `answer_with_retrieval` requires an LLM — there is nothing useful to return without one.

---

## Execution Fallbacks

These are mid-execution failures — the routing tier was selected but something failed while running. Each has a degraded-but-useful fallback rather than a 500/502.

### Intent Model Fallback (Tier 1 runtime failure)

Two cases where the `ToolAgentLoop` starts but produces no usable output:

**Case A — Loop raises an exception** (model crash, tokenizer error, OOM):
```
ToolAgentLoop.run() raises
    ↓
Log warning with exception
Re-route through Tier 2 (classify_is_search_flow if llm available, else Tier 3)
Response gains: metadata.intent_fallback = "loop_error"
```

**Case B — Loop completes but `final_answer` is empty or None** (model output no text):
```
output.final_answer is None or ""
    ↓
Same re-route as Case A
Response gains: metadata.intent_fallback = "empty_output"
```

In both cases the user gets a real answer via Tier 2/3; the failure is logged but not surfaced as an error.

---

### Search Fallback (search_tool failure)

**Inside ToolAgentLoop** (`search_tool.execute()` raises — retrieval server down):

`ToolAgentLoop` already injects a `{"role": "tool", "content": "<error message>"}` message and breaks the loop when any tool fails. The local model may or may not produce a useful `final_answer` from its own knowledge.

```
search_tool raises → ToolExecutionResult(status=FAILED)
    ↓
ToolAgentLoop breaks, returns output with final_answer (possibly from model's internal knowledge)
    ↓
if output.final_answer is not empty:
    return it — model answered from knowledge, intent = "chat", documents = []
else:
    502 — "Cannot reach retrieval server at {url}. Start it with: ..."
```

**In Tier 2/3 path** (`_run_hybrid_search()` raises):

```
_run_hybrid_search() raises
    ↓
Log warning
Fall back to answer_with_retrieval(query, llm=llm, search_url=..., top_k=0)
    → answer_with_retrieval with top_k=0 produces an LLM-only answer with no context
    → intent stays "chat", response.documents = [], citations = []
    → response gains: metadata.search_fallback = "retrieval_unavailable"
```

If `answer_with_retrieval` also fails (LLM unavailable too), that raises to the error handler → 502 with a combined message.

---

### RAG Fallback (rag_tool / answer_with_retrieval failure)

**Inside ToolAgentLoop** (`rag_tool.execute()` raises — LLM synthesis call fails):

```
rag_tool raises → ToolExecutionResult(status=FAILED)
    ↓
ToolAgentLoop breaks
    ↓
if output.final_answer is not empty:
    return it — model may have answered directly before calling rag_tool
else:
    attempt _run_direct_search(query, ...) to return raw documents
    intent = "search", answer = _search_only_answer(...)
    metadata.rag_fallback = "synthesis_failed"
```

The user gets documents even if synthesis failed. No 502 unless search also fails.

**In Tier 2/3 path** (`answer_with_retrieval()` raises — LLM call fails):

```
answer_with_retrieval() raises
    ↓
Log warning
attempt _run_direct_search(query, ...) to retrieve documents
    ↓
if documents retrieved:
    return documents + synthesized summary via _search_only_answer()
    intent = "search", metadata.rag_fallback = "synthesis_failed"
else:
    502 — "Answer generation failed and retrieval also unavailable"
```

This ensures the user sees retrieved documents rather than an error when only the synthesis step fails.

---

### Fallback Summary Table

| Failure point | Fallback action | Response to user |
|---|---|---|
| ToolAgentLoop raises | Re-route Tier 2/3 | Normal answer via LLM or rule-based search |
| ToolAgentLoop returns empty output | Re-route Tier 2/3 | Normal answer via LLM or rule-based search |
| `search_tool` fails in loop, model has answer | Return model's answer | Chat answer, no documents |
| `search_tool` fails in loop, model has no answer | 502 | "Cannot reach retrieval server at {url}" |
| `_run_hybrid_search` fails (Tier 2/3) | `answer_with_retrieval` with no context | LLM-only answer, no documents |
| `rag_tool` fails in loop, model has answer | Return model's answer | Chat answer |
| `rag_tool` fails in loop, model has no answer | `_run_direct_search` | Raw documents, no synthesis |
| `answer_with_retrieval` fails (Tier 2/3) | `_run_direct_search` | Raw documents, no synthesis |
| Both synthesis and search fail | 502 | Combined error message |

---

## Backend Changes

### 1. New Tool Registrations (`src/agents/tool_calling.py` / `src/tools/`)

Register two new `FunctionTool` wrappers at app startup alongside MCP tools:

**`search_tool`**
- Description: "Retrieve relevant documents from the corpus given a query."
- Parameters: `{ query: string, top_k?: integer }`
- Calls: `_run_direct_search(query, ...)` — the existing function in `app.py`
- Returns: JSON list of `{ title, content, url, citation }`

**`rag_tool`**
- Description: "Answer a question using retrieval-augmented generation."
- Parameters: `{ query: string, top_k?: integer }`
- Calls: `answer_with_retrieval(query, llm=llm, ...)` — already used in `chat_once`
- Returns: `{ answer: string, citations: list[string] }`

Both tools receive `search_url` and `top_k` from the request via closure at construction time (same pattern as `build_search_tool` in the existing `tool_agent` path).

### 2. `/api/agent` Routing Logic (`src/internal/servers/web/app.py`)

Replace the existing six-branch `if/elif` with:

```python
if local_model_available:
    # ToolAgentLoop with search_tool + rag_tool + MCP tools
    tools = [
        build_search_tool(search_url=search_url, top_k=top_k),
        build_rag_tool(llm=llm, search_url=search_url, top_k=top_k, filters=filters),
    ] + tool_registry.list_tools()
    output = await ToolAgentLoop(..., tools=tools).run(...)
    intent = _infer_intent_from_trajectory(output.trajectory_messages)
else:
    # Fallback: LLM-backed binary classification
    if llm and classify_is_search_flow(query, llm):
        result = await _run_hybrid_search(...)
        intent = "search"
    else:
        result = await answer_with_retrieval(...)
        intent = "chat"
```

`_infer_intent_from_trajectory` inspects `trajectory_messages` to find which tool was called first: `search_tool` → `"search"`, `rag_tool` → `"chat"`, anything else → `"tool"`.

The `mode` field in `AgentExperienceRequest` is still accepted for backwards compatibility (CLI, API clients, evals) but is ignored by the web UI frontend.

### 3. Response Shape

Add `intent` to `AgentExperienceResponse`:

```python
class AgentExperienceResponse(BaseModel):
    ...
    intent: str = "chat"  # "search" | "chat" | "tool"
```

### 4. Error Handling

Replace the catch-all `"Agent search failed"` with specific cases:

| Condition | HTTP status | Detail |
|---|---|---|
| Local model not configured + mode requires it | 400 | "Tool agent requires a local model. Set `SEARCH_AGENT_MODEL` in .env and restart." |
| Retrieval server unreachable | 502 | "Cannot reach retrieval server at {url}. Start it with: ..." (already partially done) |
| Neither local model nor LLM configured | 400 | "No LLM or local model configured. Set `OPENAI_API_KEY` or `SEARCH_AGENT_MODEL` in .env." |
| LLM not configured (no API key) | 400 | "No LLM configured. Set `OPENAI_API_KEY` or equivalent in .env." |
| Any other exception | 502 | The exception message directly, not "Agent search failed" |

---

## Frontend Changes

### 1. Remove Mode Selector (`web/src/components/SearchComposer.tsx`)

- Remove the `MODE_OPTIONS` dropdown and `Entry Point` label entirely.
- Keep: retrieval URL field, top-k field, source provider selector (for power users who set it).
- The submit button label stays "Search" but could optionally update to "Ask" — out of scope for now.

### 2. Adaptive Result Rendering (`web/src/App.tsx`)

Use `response.intent` to drive which panel is visually prominent:

| `intent` | Primary panel | Secondary panel |
|---|---|---|
| `"search"` | `SourceGrid` (full width) | `AnswerPanel` (collapsed / summary only) |
| `"chat"` | `AnswerPanel` (full width) | `SourceGrid` (collapsed) |
| `"tool"` | Tool trajectory view | `SourceGrid` if documents present |

No new components needed — `AnswerPanel` and `SourceGrid` already exist. CSS class toggling on the `results-layout` div is sufficient (`intent-search`, `intent-chat`, `intent-tool`).

### 3. Intent Badge

Show a small pill below the answer indicating what ran: `Searched · 5 sources`, `Answered · 3 citations`, `Tool · 2 calls`. Derived from `response.intent`, `response.documents.length`, and `response.citations.length`. Inline in `AnswerPanel`.

### 4. `AgentMode` Type

Remove `"search_agent"` and `"tool_agent"` from the exported `AgentMode` union in `types.ts` — they are now internal backend details. Keep `"search_tool" | "hybrid_search" | "chat_once" | "chat_loop"` for API compatibility.

---

## MCP Integration

`tool_registry.list_tools()` already returns tools registered via `/admin/tools/openapi`. These flow into `ToolAgentLoop` unchanged. No new work here — MCP tools participate automatically once the local model is wired as the router.

The admin `ToolPanel` already shows registered tools. No frontend changes needed for MCP discovery.

---

## What Does NOT Change

- CLI (`examples/run_agentic_search.py`) — still accepts explicit `--mode` flags.
- `ToolAgentLoop` itself — no changes to the loop logic.
- `AgenticRAGLoop`, `SearchAgentLoop`, `PlainGenerationLoop` — all retained for CLI/training.
- Existing `/api/agent` `mode` parameter — still accepted, overrides routing for API clients.
- All existing tests — none of the internal agent logic changes.

---

## Out of Scope

- Training the `IntentPipeline` model with search/chat/tool labels — it stays as a standalone training artifact.
- Streaming progress events for `search_tool` / `rag_tool` calls inside `ToolAgentLoop` — existing SSE stream is sufficient.
- Changing the session or history model.

---

## Testing

### Frontend — Vitest + React Testing Library

All frontend tests live in `web/src/components/__tests__/`. Pattern matches existing tests: `render()`, `screen`, `userEvent`, `vi.fn()`.

#### `SearchComposer.test.tsx` — update existing

The existing test `"shows all six mode options"` and the `mode` prop must be removed/updated since the mode dropdown is gone.

| Test | What it asserts |
|---|---|
| `"no mode dropdown is rendered"` | `screen.queryByLabelText(/entry point/i)` is `null` |
| `"renders retrieval URL and topK fields"` | Both inputs are present without a mode selector |
| `"submit enabled when query has content"` | Unchanged from existing test |
| `"submit disabled when loading"` | Unchanged from existing test |
| `"Cmd+Enter submits the form"` | `userEvent.keyboard('{Meta>}{Enter}{/Meta}')` triggers `onSubmit` |

The `mode` and `onModeChange` props are removed from `SearchComposerProps`; no test should reference them.

#### `AnswerPanel.test.tsx` — update existing + new intent badge tests

`AnswerPanel` gains an optional `intent` and `documentCount` prop for the badge.

| Test | What it asserts |
|---|---|
| `"renders 'Searched · N sources' badge when intent is search"` | `screen.getByText(/searched · 5 sources/i)` present with `intent="search" documentCount={5}` |
| `"renders 'Answered · N citations' badge when intent is chat"` | `screen.getByText(/answered · 2 citations/i)` present with `intent="chat"` and 2 citations |
| `"renders 'Used tools · N calls' badge when intent is tool"` | `screen.getByText(/used tools/i)` present with `intent="tool" toolCallCount={2}` |
| `"renders no badge when intent is undefined"` | No `.intent-badge` element in document |
| `"renders no badge when answer is empty"` | Empty state shown; no badge |

#### `App.test.tsx` — new file

Integration-level tests for the `App` component using `vi.mock("./api")` to stub `runAgent` and `streamAgent`.

| Test | Setup | What it asserts |
|---|---|---|
| `"shows source grid prominently on search intent response"` | `runAgent` resolves with `intent: "search"`, 3 documents | Results layout has `intent-search` class; source grid section is not collapsed |
| `"shows answer panel prominently on chat intent response"` | `runAgent` resolves with `intent: "chat"`, answer text | Results layout has `intent-chat` class; answer panel is not collapsed |
| `"shows tool trajectory panel on tool intent response"` | `runAgent` resolves with `intent: "tool"`, 0 documents | Results layout has `intent-tool` class |
| `"shows error banner on API failure"` | `runAgent` rejects with `new Error("No LLM configured")` | `screen.getByText(/no llm configured/i)` is present |
| `"clears error when new session is created"` | Error state then click "New" button | Error banner gone |
| `"new session button resets answer, documents, citations"` | Populated state → click New | `answer`, `documents`, `citations` all cleared |

#### `api.test.ts` — new file (unit, no DOM)

| Test | What it asserts |
|---|---|
| `"runAgent passes intent field through from response"` | Mocked `fetch` returning `{ intent: "search", ... }` → resolved value has `.intent === "search"` |
| `"streamAgent yields error event on non-ok response"` | 502 response → generator yields `{ type: "error", detail: "..." }` |

---

### Backend — pytest unit tests

#### `tests/unit/servers/web/test_web_experience_app.py` — add to existing

| Test | What it asserts |
|---|---|
| `test_agent_endpoint_returns_intent_field` | `/api/agent` response JSON has `"intent"` key (value `"chat"` in default fallback path) |
| `test_agent_endpoint_explicit_mode_overrides_routing` | `mode="search_tool"` in request → `answer_with_retrieval` is NOT called; search path runs |
| `test_agent_no_llm_no_model_returns_400` | App with no LLM and no local model → POST `/api/agent` → 400, detail contains "No LLM or local model configured" |
| `test_agent_tool_mode_without_model_returns_clear_400` | `mode="tool_agent"` with no `search_agent_manager` → 400, detail contains "local model" |
| `test_agent_retrieval_server_down_returns_502_with_url` | `answer_with_retrieval` raises `ConnectionError` → 502, detail contains the search URL |
| `test_agent_other_exception_returns_502_with_message` | `answer_with_retrieval` raises `ValueError("bad input")` → 502, detail is `"bad input"`, NOT `"Agent search failed"` |

#### `tests/unit/test_intent_routing.py` — new file

Tests for `_infer_intent_from_trajectory` and `build_search_tool` / `build_rag_tool`.

| Test | What it asserts |
|---|---|
| `test_infer_intent_search_tool_call` | Trajectory with `{"role": "tool_call", "name": "search_tool"}` → `"search"` |
| `test_infer_intent_rag_tool_call` | Trajectory with `{"role": "tool_call", "name": "rag_tool"}` → `"chat"` |
| `test_infer_intent_mcp_tool_call` | Trajectory with `{"role": "tool_call", "name": "custom_api"}` → `"tool"` |
| `test_infer_intent_no_tool_calls` | Trajectory with only assistant messages → `"chat"` (default) |
| `test_build_search_tool_schema` | `build_search_tool(search_url=..., top_k=5).schema.to_dict()` has `"query"` parameter |
| `test_build_rag_tool_schema` | `build_rag_tool(llm=..., search_url=..., top_k=5).schema.to_dict()` has `"query"` parameter |
| `test_search_tool_executes_returns_json_string` | `await search_tool.execute("default", {"query": "FAISS"})` with mocked `_run_direct_search` → returns JSON string |
| `test_rag_tool_executes_returns_answer` | `await rag_tool.execute("default", {"query": "What is FAISS?"})` with mocked `answer_with_retrieval` → answer string |

#### `tests/unit/test_secondary_llm_flows.py` — extend existing

| Test | What it asserts |
|---|---|
| `test_classify_is_search_flow_returns_true_for_search_query` | Already exists — keep |
| `test_classify_fallback_routing_used_when_no_local_model` | With `llm` set and no `search_agent_manager`, POST `/api/agent` calls `classify_is_search_flow` |

#### `tests/unit/test_intent_routing.py` — fallback chain additions

| Test | What it asserts |
|---|---|
| `test_rule_based_is_search_find_keyword` | `_rule_based_is_search("find me the onboarding doc")` → `True` |
| `test_rule_based_is_search_short_keyword_query` | `_rule_based_is_search("procurement process")` → `True` (≤3 tokens, no verb) |
| `test_rule_based_is_chat_explain_keyword` | `_rule_based_is_search("explain how FAISS works")` → `False` |
| `test_rule_based_is_chat_default` | `_rule_based_is_search("what led us to win the deal")` → `False` |
| `test_llm_classifier_exception_falls_back_to_rule_based` | `classify_is_search_flow` raises `RuntimeError` → routing still resolves via rule-based, no 500 |
| `test_tier3_only_search_query_runs_hybrid_search` | No LLM, no local model, search-intent query → `_run_hybrid_search` called, `intent="search"` |
| `test_tier3_only_chat_query_returns_400` | No LLM, no local model, chat-intent query → 400, detail contains "No LLM configured" |
| `test_explicit_tool_mode_without_local_model_returns_400` | `mode="tool_agent"`, no `search_agent_manager` → 400, detail contains "local model" |

#### `tests/unit/test_execution_fallbacks.py` — new file

Tests for mid-execution fallbacks using `monkeypatch` to simulate failures.

**Intent model fallbacks:**

| Test | Setup | What it asserts |
|---|---|---|
| `test_tool_loop_raises_reroutes_to_tier2` | `ToolAgentLoop.run` raises `RuntimeError`; `llm` configured; `classify_is_search_flow` returns `False` | Response 200, `intent="chat"`, `metadata["intent_fallback"] == "loop_error"` |
| `test_tool_loop_empty_output_reroutes_to_tier2` | `ToolAgentLoop.run` returns `output` with `final_answer=None`; `llm` configured | Response 200, `intent="chat"`, `metadata["intent_fallback"] == "empty_output"` |
| `test_tool_loop_raises_reroutes_to_tier3_when_no_llm` | `ToolAgentLoop.run` raises; no `llm`; search-intent query | Response 200, `intent="search"`, `_run_hybrid_search` called |

**Search fallbacks:**

| Test | Setup | What it asserts |
|---|---|---|
| `test_search_tool_failure_model_has_answer` | `search_tool.execute` raises; `ToolAgentLoop` returns `final_answer="answered from memory"` | Response 200, `answer="answered from memory"`, `documents=[]`, `intent="chat"` |
| `test_search_tool_failure_model_no_answer` | `search_tool.execute` raises; `ToolAgentLoop` returns `final_answer=None` | Response 502, detail contains retrieval server URL |
| `test_hybrid_search_failure_falls_back_to_rag` | `_run_hybrid_search` raises `ConnectionError`; `answer_with_retrieval` succeeds | Response 200, `intent="chat"`, `metadata["search_fallback"] == "retrieval_unavailable"`, `documents=[]` |
| `test_hybrid_search_and_rag_both_fail_returns_502` | Both `_run_hybrid_search` and `answer_with_retrieval` raise | Response 502, detail is combined error message |

**RAG fallbacks:**

| Test | Setup | What it asserts |
|---|---|---|
| `test_rag_tool_failure_model_has_answer` | `rag_tool.execute` raises; `ToolAgentLoop` returns `final_answer="direct answer"` | Response 200, `answer="direct answer"`, `intent="chat"` |
| `test_rag_tool_failure_model_no_answer_falls_back_to_search` | `rag_tool.execute` raises; `ToolAgentLoop` returns `final_answer=None`; `_run_direct_search` succeeds with 2 docs | Response 200, `intent="search"`, `documents` has 2 entries, `metadata["rag_fallback"] == "synthesis_failed"` |
| `test_answer_with_retrieval_failure_falls_back_to_raw_documents` | `answer_with_retrieval` raises; `_run_direct_search` returns 3 docs | Response 200, `intent="search"`, `documents` has 3 entries, `metadata["rag_fallback"] == "synthesis_failed"` |
| `test_answer_with_retrieval_and_search_both_fail_returns_502` | Both `answer_with_retrieval` and `_run_direct_search` raise | Response 502, detail contains "retrieval also unavailable" |

---

### What is NOT tested here

- `ToolAgentLoop` internals — tested in existing `tests/unit/test_agent_loop.py`
- `classify_is_search_flow` prompt correctness — tested in existing `test_secondary_llm_flows.py`
- MCP tool registration — tested in existing `test_mcp_server.py`
- Session persistence — tested in existing `test_agent_endpoint_reuses_existing_session_history`

---

## Success Criteria

1. A user can type a query and get a sensible result without selecting a mode.
2. "Agent search failed" never appears — all errors have actionable detail.
3. `response.intent` correctly reflects which path ran.
4. The frontend renders the primary panel for the right intent.
5. Existing CLI and API clients (`mode` param) continue to work unchanged.
6. All existing tests pass.
