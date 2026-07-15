# Generated Context Pack

# Intent Routed Search Chat

## Sources

- [Specification: 2026-06-15-intent-routed-search-chat-design.md](../specs/2026-06-15-intent-routed-search-chat-design.md)
- [Plan: 2026-06-15-intent-routed-search-chat.md](../plans/2026-06-15-intent-routed-search-chat.md)

## Specification Context

### Goal

- Single input box — no mode selector visible to users.
- The system routes automatically: the `ToolAgentLoop` (local model) acts as the universal router when available, expressing intent by which tool it calls.
- Clean fallback when no local model is configured.
- Errors are specific and actionable, never "Agent search failed".

---

### 1. Remove Mode Selector (`web/src/components/SearchComposer.tsx`)

- Remove the `MODE_OPTIONS` dropdown and `Entry Point` label entirely.
- Keep: retrieval URL field, top-k field, source provider selector (for power users who set it).
- The submit button label stays "Search" but could optionally update to "Ask" — out of scope for now.

### Out of Scope

- Training the `IntentPipeline` model with search/chat/tool labels — it stays as a standalone training artifact.
- Streaming progress events for `search_tool` / `rag_tool` calls inside `ToolAgentLoop` — existing SSE stream is sufficient.
- Changing the session or history model.

---

### Frontend — Vitest + React Testing Library

All frontend tests live in `web/src/components/__tests__/`. Pattern matches existing tests: `render()`, `screen`, `userEvent`, `vi.fn()`.

### `SearchComposer.test.tsx` — update existing

The existing test `"shows all six mode options"` and the `mode` prop must be removed/updated since the mode dropdown is gone.

| Test | What it asserts |
|---|---|
| `"no mode dropdown is rendered"` | `screen.queryByLabelText(/entry point/i)` is `null` |
| `"renders retrieval URL and topK fields"` | Both inputs are present without a mode selector |
| `"submit enabled when query has content"` | Unchanged from existing test |
| `"submit disabled when loading"` | Unchanged from existing test |
| `"Cmd+Enter submits the form"` | `userEvent.keyboard('{Meta>}{Enter}{/Meta}')` triggers `onSubmit` |

The `mode` and `onModeChange` props are removed from `SearchComposerProps`; no test should reference them.

### `AnswerPanel.test.tsx` — update existing + new intent badge tests

`AnswerPanel` gains an optional `intent` and `documentCount` prop for the badge.

| Test | What it asserts |
|---|---|
| `"renders 'Searched · N sources' badge when intent is search"` | `screen.getByText(/searched · 5 sources/i)` present with `intent="search" documentCount={5}` |
| `"renders 'Answered · N citations' badge when intent is chat"` | `screen.getByText(/answered · 2 citations/i)` present with `intent="chat"` and 2 citations |
| `"renders 'Used tools · N calls' badge when intent is tool"` | `screen.getByText(/used tools/i)` present with `intent="tool" toolCallCount={2}` |
| `"renders no badge when intent is undefined"` | No `.intent-badge` element in document |
| `"renders no badge when answer is empty"` | Empty state shown; no badge |

### `App.test.tsx` — new file

Integration-level tests for the `App` component using `vi.mock("./api")` to stub `runAgent` and `streamAgent`.

| Test | Setup | What it asserts |
|---|---|---|
| `"shows source grid prominently on search intent response"` | `runAgent` resolves with `intent: "search"`, 3 documents | Results layout has `intent-search` class; source grid section is not collapsed |
| `"shows answer panel prominently on chat intent response"` | `runAgent` resolves with `intent: "chat"`, answer text | Results layout has `intent-chat` class; answer panel is not collapsed |
| `"shows tool trajectory panel on tool intent response"` | `runAgent` resolves with `intent: "tool"`, 0 documents | Results layout has `intent-tool` class |
| `"shows error banner on API failure"` | `runAgent` rejects with `new Error("No LLM configured")` | `screen.getByText(/no llm configured/i)` is present |
| `"clears error when new session is created"` | Error state then click "New" button | Error banner gone |
| `"new session button resets answer, documents, citations"` | Populated state → click New | `answer`, `documents`, `citations` all cleared |

### `api.test.ts` — new file (unit, no DOM)

| Test | What it asserts |
|---|---|
| `"runAgent passes intent field through from response"` | Mocked `fetch` returning `{ intent: "search", ... }` → resolved value has `.intent === "search"` |
| `"streamAgent yields error event on non-ok response"` | 502 response → generator yields `{ type: "error", detail: "..." }` |

---

## Implementation Plan Context

### Task 1: Rule-based classifier + trajectory intent inference

**Files:**
- Create: `src/internal/servers/web/intent_routing.py`
- Create: `tests/unit/test_intent_routing.py`

- [ ] **Step 1: Write failing tests**

```python

### tests/unit/test_intent_routing.py

import json
import pytest
from src.internal.servers.web.intent_routing import (
    _rule_based_is_search,
    _infer_intent_from_output,
)
from src.agents.base import AgentLoopOutput


def _make_output(action_trace: str | None = None, final_answer: str | None = None) -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        action_trace=action_trace,
        final_answer=final_answer,
    )

### Task 2: Routing tool builders

**Files:**
- Create: `src/tools/routing_tools.py`
- Modify: `src/tools/__init__.py`
- Test: `tests/unit/test_intent_routing.py` (append)

- [ ] **Step 1: Write failing tests (append to test file)**

```python

### Append to tests/unit/test_intent_routing.py

from src.tools.routing_tools import build_search_routing_tool, build_rag_routing_tool


def test_build_search_routing_tool_schema():
    tool = build_search_routing_tool(search_url="http://localhost:8000/retrieve", top_k=5)
    schema = tool.schema.to_dict()
    assert schema["function"]["name"] == "search_routing_tool"
    assert "query" in schema["function"]["parameters"]["properties"]


def test_build_rag_routing_tool_schema():
    tool = build_rag_routing_tool(llm=None, search_url="http://localhost:8000/retrieve", top_k=5)
    schema = tool.schema.to_dict()
    assert schema["function"]["name"] == "rag_routing_tool"
    assert "query" in schema["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_search_routing_tool_returns_json(monkeypatch):
    from src.tools import SearchPage
    async def fake_search_tool(query, *, provider, search_url, page_size):
        return [SearchPage(title="Doc A", summary="summary", url="http://example.com", error=None)]
    monkeypatch.setattr("src.tools.routing_tools.search_tool", fake_search_tool)
    tool = build_search_routing_tool(search_url="http://localhost:8000/retrieve", top_k=5)
    result, _, _ = await tool.execute("default", {"query": "FAISS"})
    data = json.loads(result)
    assert data[0]["title"] == "Doc A"


@pytest.mark.asyncio
async def test_rag_routing_tool_returns_answer(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    from src.context.models import AnswerGenerationResult, SearchContextBundle, PromptBundle
    fake_result = AnswerGenerationResult(

_[Section compacted.]_

### Task 3: Update request/response models

**Files:**
- Modify: `src/internal/servers/web/app.py` (models only — lines 148–176)
- Test: `tests/unit/servers/web/test_web_experience_app.py` (append)

- [ ] **Step 1: Write failing test (append to existing test file)**

```python

### Append to tests/unit/servers/web/test_web_experience_app.py

def test_agent_endpoint_returns_intent_field(monkeypatch, tmp_path):
    async def fake_answer(*args, **kwargs):
        return _answer_result("q")
    monkeypatch.setattr("src.internal.servers.web.app.answer_with_retrieval", fake_answer)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "explain FAISS"})
    assert response.status_code == 200
    data = response.json()
    assert "intent" in data
    assert data["intent"] in ("search", "chat", "tool")
```

- [ ] **Step 2: Run test — expect KeyError (field missing)**

```bash
pytest tests/unit/servers/web/test_web_experience_app.py::test_agent_endpoint_returns_intent_field -v
```

Expected: FAIL — `"intent" not in data`

- [ ] **Step 3: Update models in `app.py`**

Change `AgentExperienceRequest.mode` (around line 161):
```python

### Task 4: Auto-routing dispatch (`_run_auto_routed`)

**Files:**
- Modify: `src/internal/servers/web/app.py` (add `_run_auto_routed` and wire into `run_agent`)
- Test: `tests/unit/servers/web/test_web_experience_app.py` (append)

- [ ] **Step 1: Write failing tests**

```python

### Append to tests/unit/servers/web/test_web_experience_app.py

def test_auto_route_chat_uses_answer_with_retrieval(monkeypatch, tmp_path):
    """No mode in request → auto-routes to chat via answer_with_retrieval."""
    called = {}
    async def fake_answer(q, *, llm, chat_history, search_url, top_k, filters):
        called["answer"] = True
        return _answer_result(q)
    monkeypatch.setattr("src.internal.servers.web.app.answer_with_retrieval", fake_answer)
    # classify_is_search_flow not called when no llm configured
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "explain FAISS"})
    assert response.status_code == 200
    assert called.get("answer") is True
    assert response.json()["intent"] == "chat"


def test_auto_route_search_via_rule_based(monkeypatch, tmp_path):
    """Short keyword query → rule-based classifies as search → hybrid_search runs."""
    called = {}
    async def fake_hybrid(query, *, llm, search_url, browser_search_url, rerank_url, top_k, filters, source_provider):
        called["hybrid"] = True
        from src.internal.servers.web.app import _HybridSearchResult
        return _HybridSearchResult(executed_queries=[query], documents=[])
    monkeypatch.setattr("src.internal.servers.web.app._run_hybrid_search", fake_hybrid)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "procurement process"})
    assert response.status_code == 200

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
