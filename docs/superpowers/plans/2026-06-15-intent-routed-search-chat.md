# Intent-Routed Search & Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the manual mode-selector UX with automatic intent routing — ToolAgentLoop as the universal router when a local model is available, with LLM-backed and rule-based fallbacks, execution-level graceful degradation, and adaptive frontend rendering based on `response.intent`.

**Architecture:** A new `intent_routing.py` module provides the rule-based classifier and trajectory intent inference. Two new `FunctionTool` wrappers (`search_routing_tool`, `rag_routing_tool`) register search and RAG as callable tools. The `/api/agent` endpoint auto-routes when `mode` is `None`, using a three-tier cascade: ToolAgentLoop → LLM classifier → rule-based. Frontend drops the mode dropdown and adapts layout to `response.intent`.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, `FunctionTool` from `src/tools/base.py`, React 19 + TypeScript, Vitest + React Testing Library, pytest + `monkeypatch`.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| **Create** | `src/internal/servers/web/intent_routing.py` | `_rule_based_is_search`, `_infer_intent_from_output` |
| **Create** | `src/tools/routing_tools.py` | `build_search_routing_tool`, `build_rag_routing_tool` |
| **Create** | `tests/unit/test_intent_routing.py` | All routing + fallback unit tests |
| **Create** | `tests/unit/test_execution_fallbacks.py` | Mid-execution fallback tests |
| **Create** | `web/src/components/__tests__/App.test.tsx` | App adaptive layout tests |
| **Create** | `web/src/__tests__/api.test.ts` | API response shape tests |
| **Modify** | `src/internal/servers/web/app.py` | Add `intent` field, replace six-branch dispatch, fix errors |
| **Modify** | `src/tools/__init__.py` | Export new routing tools |
| **Modify** | `web/src/types.ts` | Add `intent` to response, trim `AgentMode` |
| **Modify** | `web/src/components/SearchComposer.tsx` | Remove mode dropdown + props |
| **Modify** | `web/src/components/AnswerPanel.tsx` | Add intent badge |
| **Modify** | `web/src/App.tsx` | Adaptive layout class, remove mode state |
| **Modify** | `web/src/components/__tests__/SearchComposer.test.tsx` | Remove mode tests |
| **Modify** | `web/src/components/__tests__/AnswerPanel.test.tsx` | Add badge tests |
| **Modify** | `tests/unit/servers/web/test_web_experience_app.py` | Add intent/error tests |

---

## Task 1: Rule-based classifier + trajectory intent inference

**Files:**
- Create: `src/internal/servers/web/intent_routing.py`
- Create: `tests/unit/test_intent_routing.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_intent_routing.py
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


# --- _rule_based_is_search ---

def test_rule_based_is_search_find_keyword():
    assert _rule_based_is_search("find me the onboarding doc") is True

def test_rule_based_is_search_short_keyword_query():
    assert _rule_based_is_search("procurement process") is True  # ≤5 tokens, no verb

def test_rule_based_is_search_list_keyword():
    assert _rule_based_is_search("list all pull requests since last week") is True

def test_rule_based_is_chat_explain_keyword():
    assert _rule_based_is_search("explain how FAISS works") is False

def test_rule_based_is_chat_what_is():
    assert _rule_based_is_search("what is the difference between BM25 and dense retrieval") is False

def test_rule_based_is_chat_default_no_signal():
    assert _rule_based_is_search("what led us to win the deal with company X") is False

def test_rule_based_is_search_show_me():
    assert _rule_based_is_search("show me the deployment runbook") is True

# --- _infer_intent_from_output ---

def _trace(tool_name: str) -> str:
    return json.dumps({"tool_name": tool_name, "status": "completed", "result": "ok",
                       "performance": {}, "error_code": None, "error_message": None,
                       "optimization_suggestions": [], "retry_count": 0})

def test_infer_intent_search_routing_tool():
    output = _make_output(action_trace=_trace("search_routing_tool"))
    assert _infer_intent_from_output(output) == "search"

def test_infer_intent_rag_routing_tool():
    output = _make_output(action_trace=_trace("rag_routing_tool"))
    assert _infer_intent_from_output(output) == "chat"

def test_infer_intent_mcp_tool():
    output = _make_output(action_trace=_trace("custom_api"))
    assert _infer_intent_from_output(output) == "tool"

def test_infer_intent_no_trace_defaults_to_chat():
    output = _make_output(action_trace=None)
    assert _infer_intent_from_output(output) == "chat"

def test_infer_intent_malformed_trace_defaults_to_chat():
    output = _make_output(action_trace="not json")
    assert _infer_intent_from_output(output) == "chat"
```

- [ ] **Step 2: Run tests — expect ImportError (module doesn't exist)**

```bash
pytest tests/unit/test_intent_routing.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.internal.servers.web.intent_routing'`

- [ ] **Step 3: Create `intent_routing.py`**

```python
# src/internal/servers/web/intent_routing.py
"""Intent routing helpers for the /api/agent endpoint."""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agents.base import AgentLoopOutput

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


def _rule_based_is_search(query: str) -> bool:
    """Return True if the query looks like a search/retrieval intent."""
    q = query.strip()
    if _SEARCH_RE.search(q):
        return True
    tokens = q.split()
    if len(tokens) <= 5 and not _VERB_RE.search(q) and not q.endswith("?"):
        return True
    if _CHAT_RE.search(q):
        return False
    return False


def _infer_intent_from_output(output: "AgentLoopOutput") -> str:
    """Infer search/chat/tool intent from the first tool called in the output."""
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
    except (json.JSONDecodeError, KeyError, AttributeError):
        pass
    return "chat"
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/unit/test_intent_routing.py -v
```

Expected: 12 tests pass.

- [ ] **Step 5: Commit**

```bash
git checkout -b feat/intent-routed-search-chat
git add src/internal/servers/web/intent_routing.py tests/unit/test_intent_routing.py
git commit -m "feat: add rule-based classifier and trajectory intent inference"
```

---

## Task 2: Routing tool builders

**Files:**
- Create: `src/tools/routing_tools.py`
- Modify: `src/tools/__init__.py`
- Test: `tests/unit/test_intent_routing.py` (append)

- [ ] **Step 1: Write failing tests (append to test file)**

```python
# Append to tests/unit/test_intent_routing.py

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
        answer="FAISS is a library.",
        citations=["[D1]"],
        context=SearchContextBundle(query="q", documents=[]),
        prompt=PromptBundle(system="", user="", messages=[]),
    )
    mock_llm = MagicMock()
    monkeypatch.setattr("src.tools.routing_tools.answer_with_retrieval", AsyncMock(return_value=fake_result))
    tool = build_rag_routing_tool(llm=mock_llm, search_url="http://localhost:8000/retrieve", top_k=5)
    result, _, _ = await tool.execute("default", {"query": "What is FAISS?"})
    data = json.loads(result)
    assert data["answer"] == "FAISS is a library."
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
pytest tests/unit/test_intent_routing.py -v -k "routing_tool"
```

Expected: `ModuleNotFoundError: No module named 'src.tools.routing_tools'`

- [ ] **Step 3: Create `routing_tools.py`**

```python
# src/tools/routing_tools.py
"""FunctionTool wrappers for search and RAG, used as ToolAgentLoop routing tools."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .base import FunctionTool
from .search import search_tool

if TYPE_CHECKING:
    from src.context.models import LLMClient, SearchFilters

_SEARCH_TOOL_PARAMS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query."},
    },
    "required": ["query"],
}

_RAG_TOOL_PARAMS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The question to answer using retrieval."},
    },
    "required": ["query"],
}


def build_search_routing_tool(*, search_url: str, top_k: int) -> FunctionTool:
    """FunctionTool that retrieves documents from the corpus."""

    async def _execute(query: str) -> str:
        pages = await search_tool(
            query,
            provider="retrieval",
            search_url=search_url,
            page_size=top_k,
        )
        results = [
            {"title": p.title or "", "content": p.summary or "", "url": p.url}
            for p in pages
            if not p.error
        ]
        return json.dumps(results)

    return FunctionTool(
        fn=_execute,
        name="search_routing_tool",
        description="Retrieve relevant documents from the corpus given a search query.",
        parameters=_SEARCH_TOOL_PARAMS,
    )


def build_rag_routing_tool(
    *,
    llm: "LLMClient | None",
    search_url: str,
    top_k: int,
    filters: "SearchFilters | None" = None,
) -> FunctionTool:
    """FunctionTool that generates a RAG answer."""
    from src.context import answer_with_retrieval

    async def _execute(query: str) -> str:
        result = await answer_with_retrieval(
            query,
            llm=llm,
            search_url=search_url,
            top_k=top_k,
            filters=filters,
        )
        return json.dumps({"answer": result.answer, "citations": result.citations})

    return FunctionTool(
        fn=_execute,
        name="rag_routing_tool",
        description="Answer a question using retrieval-augmented generation.",
        parameters=_RAG_TOOL_PARAMS,
    )
```

- [ ] **Step 4: Export from `src/tools/__init__.py`**

Add these two lines at the end of `src/tools/__init__.py`:

```python
from .routing_tools import build_search_routing_tool as build_search_routing_tool
from .routing_tools import build_rag_routing_tool as build_rag_routing_tool
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest tests/unit/test_intent_routing.py -v -k "routing_tool"
```

Expected: 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/tools/routing_tools.py src/tools/__init__.py tests/unit/test_intent_routing.py
git commit -m "feat: add search_routing_tool and rag_routing_tool FunctionTool wrappers"
```

---

## Task 3: Update request/response models

**Files:**
- Modify: `src/internal/servers/web/app.py` (models only — lines 148–176)
- Test: `tests/unit/servers/web/test_web_experience_app.py` (append)

- [ ] **Step 1: Write failing test (append to existing test file)**

```python
# Append to tests/unit/servers/web/test_web_experience_app.py

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
# Before:
mode: str = Field(
    default="chat_once",
    description=(...),
)

# After:
mode: str | None = Field(
    default=None,
    description=(
        "Optional explicit mode override: 'search_tool', 'hybrid_search', 'chat_once', "
        "'chat_loop', 'search_agent', 'tool_agent'. When None, intent is auto-detected."
    ),
)
```

Add `intent` to `AgentExperienceResponse` (around line 171, after `hook_metadata`):
```python
class AgentExperienceResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[str]
    documents: list[SourceDocumentView]
    messages: list[ChatMessageView]
    hook_metadata: dict[str, object] = Field(default_factory=dict)
    intent: str = "chat"  # "search" | "chat" | "tool"
```

- [ ] **Step 4: Run test — expect PASS**

```bash
pytest tests/unit/servers/web/test_web_experience_app.py::test_agent_endpoint_returns_intent_field -v
```

Expected: PASS.

- [ ] **Step 5: Run full existing test suite to catch regressions**

```bash
pytest tests/unit/servers/web/test_web_experience_app.py -v
```

Expected: all existing tests still pass (they don't check `intent` was absent).

- [ ] **Step 6: Commit**

```bash
git add src/internal/servers/web/app.py tests/unit/servers/web/test_web_experience_app.py
git commit -m "feat: add intent field to AgentExperienceResponse, make mode optional"
```

---

## Task 4: Auto-routing dispatch (`_run_auto_routed`)

**Files:**
- Modify: `src/internal/servers/web/app.py` (add `_run_auto_routed` and wire into `run_agent`)
- Test: `tests/unit/servers/web/test_web_experience_app.py` (append)

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/unit/servers/web/test_web_experience_app.py

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
    assert called.get("hybrid") is True
    assert response.json()["intent"] == "search"


def test_explicit_mode_still_works(monkeypatch, tmp_path):
    """Passing explicit mode='chat_once' still routes to answer_with_retrieval."""
    called = {}
    async def fake_answer(q, *, llm, chat_history, search_url, top_k, filters):
        called["answer"] = True
        return _answer_result(q)
    monkeypatch.setattr("src.internal.servers.web.app.answer_with_retrieval", fake_answer)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "hello", "mode": "chat_once"})
    assert response.status_code == 200
    assert called.get("answer") is True
```

- [ ] **Step 2: Run tests — expect FAIL (auto-routing not implemented)**

```bash
pytest tests/unit/servers/web/test_web_experience_app.py -k "auto_route or explicit_mode" -v
```

Expected: `test_auto_route_chat_uses_answer_with_retrieval` likely passes (chat_once is default), others may fail or pass for wrong reasons.

- [ ] **Step 3: Add `_run_auto_routed` to `app.py`**

Add this import at top of `app.py`:
```python
from src.internal.servers.web.intent_routing import _rule_based_is_search, _infer_intent_from_output
from src.internal.servers.secondary_llm_flows.search_flow_classification import classify_is_search_flow
from src.tools.routing_tools import build_search_routing_tool, build_rag_routing_tool
```

Add this function before `run_agent` in `app.py` (around line 410):

```python
async def _run_auto_routed(
    query: str,
    *,
    llm: LLMClient | None,
    manager: object | None,
    tokenizer: object | None,
    search_url: str,
    browser_search_url: str | None,
    rerank_url: str | None,
    top_k: int,
    filters: SearchFilters | None,
    history: list[ChatMessage],
    resolved: AppSettings,
) -> tuple[str, list[str], list[ContextDocument], str, dict[str, str]]:
    """
    Returns (answer, citations, documents, intent, fallback_meta).
    Tries Tier 1 (ToolAgentLoop) → Tier 2 (LLM classify) → Tier 3 (rule-based).
    """
    extra: dict[str, str] = {}

    # --- Tier 1: ToolAgentLoop ---
    if manager is not None and tokenizer is not None:
        from src.agents.tool_calling import ToolAgentLoop, ToolAgentLoopConfig
        tools = [
            build_search_routing_tool(search_url=search_url, top_k=top_k),
            build_rag_routing_tool(llm=llm, search_url=search_url, top_k=top_k, filters=filters),
        ] + tool_registry.list_tools()
        loop = ToolAgentLoop(
            tokenizer=tokenizer,
            server_manager=manager,
            tools=tools,
            config=ToolAgentLoopConfig(tool_parser_format=resolved.tool_agent_parser),
        )
        try:
            output = await loop.run(
                [{"role": "user", "content": query}],
                sampling_params={"temperature": 0.0, "max_tokens": 512},
            )
        except Exception as exc:
            logger.warning("ToolAgentLoop failed, falling through to Tier 2: %s", exc)
            extra["intent_fallback"] = "loop_error"
            output = None

        if output is not None and not (output.final_answer or "").strip():
            logger.warning("ToolAgentLoop returned empty output, falling through to Tier 2")
            extra["intent_fallback"] = "empty_output"
            output = None

        if output is not None:
            intent = _infer_intent_from_output(output)
            answer = output.final_answer or ""
            # Extract documents from search_routing_tool result if present
            documents: list[ContextDocument] = []
            if output.action_trace:
                import json as _json
                for line in output.action_trace.split("\n"):
                    try:
                        rec = _json.loads(line)
                        if rec.get("tool_name") == "search_routing_tool" and rec.get("result"):
                            raw = _json.loads(rec["result"])
                            for i, item in enumerate(raw, 1):
                                documents.append(ContextDocument(
                                    id=f"D{i}", title=item.get("title", ""),
                                    content=item.get("content", ""), url=item.get("url"),
                                    score=0.0, metadata={"source": "search_routing_tool"},
                                ))
                    except Exception:
                        pass
            citations = [doc.citation for doc in documents]
            return answer, citations, documents, intent, extra

    # --- Tier 2: LLM classify + execution with fallbacks ---
    if llm is not None:
        try:
            is_search = classify_is_search_flow(query, llm)
        except Exception as exc:
            logger.warning("LLM classifier failed, using rule-based: %s", exc)
            is_search = _rule_based_is_search(query)
    else:
        # --- Tier 3: rule-based only ---
        is_search = _rule_based_is_search(query)
        if not is_search:
            raise HTTPException(
                status_code=400,
                detail="No LLM configured. Set OPENAI_API_KEY or equivalent in .env.",
            )

    if is_search:
        try:
            search_result = await _run_hybrid_search(
                query,
                llm=llm,
                search_url=search_url,
                browser_search_url=browser_search_url,
                rerank_url=rerank_url,
                top_k=top_k,
                filters=filters,
                source_provider="retrieval",
            )
            answer = _search_only_answer(
                "Search", queries=search_result.executed_queries,
                documents=search_result.documents, source_provider="retrieval",
            )
            return answer, [d.citation for d in search_result.documents], search_result.documents, "search", extra
        except Exception as exc:
            logger.warning("Hybrid search failed, falling back to RAG without context: %s", exc)
            extra["search_fallback"] = "retrieval_unavailable"
            # Fall through to chat path below
            is_search = False

    # Chat path (also the search fallback path)
    if llm is None:
        raise HTTPException(
            status_code=400,
            detail="No LLM or local model configured. Set OPENAI_API_KEY or SEARCH_AGENT_MODEL in .env.",
        )
    try:
        result = await answer_with_retrieval(
            query, llm=llm, chat_history=history,
            search_url=search_url, top_k=0 if extra.get("search_fallback") else top_k,
            filters=filters,
        )
        return (
            result.answer,
            result.citations,
            result.context.documents,
            "chat",
            extra,
        )
    except Exception as exc:
        logger.warning("RAG answer_with_retrieval failed, trying raw search: %s", exc)
        extra["rag_fallback"] = "synthesis_failed"
        try:
            raw_docs = await _run_direct_search(
                query, source_provider="retrieval",
                search_url=search_url, browser_search_url=None,
                rerank_url=None, top_k=top_k,
            )
            if raw_docs:
                answer = _search_only_answer(
                    "Search (synthesis failed)", queries=[query],
                    documents=raw_docs, source_provider="retrieval",
                )
                return answer, [d.citation for d in raw_docs], raw_docs, "search", extra
        except Exception as exc2:
            raise HTTPException(
                status_code=502,
                detail=f"Answer generation failed and retrieval also unavailable: {exc2}",
            ) from exc2
        raise HTTPException(status_code=502, detail=str(exc)) from exc
```

- [ ] **Step 4: Wire `_run_auto_routed` into `run_agent`**

In `run_agent`, replace the existing dispatch block (the try/except starting at line 458) with:

```python
        mode_str = request.mode.strip().lower() if request.mode else None
        normalized_mode = _normalize_agent_mode(mode_str) if mode_str else None
        manager = getattr(http_request.app.state, "search_agent_manager", None)
        tokenizer = getattr(http_request.app.state, "search_agent_tokenizer", None)

        try:
            if normalized_mode is None:
                # Auto-routing (new web UI path)
                answer, citations, documents, intent, extra_meta = await _run_auto_routed(
                    query,
                    llm=llm,
                    manager=manager,
                    tokenizer=tokenizer,
                    search_url=search_url,
                    browser_search_url=settings.browser_search_url,
                    rerank_url=settings.rerank_url,
                    top_k=top_k,
                    filters=filters,
                    history=history,
                    resolved=resolved,
                )
                merged_metadata = {**hook_metadata, **extra_meta}
                db.add_chat_message(
                    session_id, role="assistant", content=answer,
                    metadata={"citations": citations,
                              "document_ids": [d.id for d in documents],
                              "hooks": hook_metadata, "mode": "auto",
                              "intent": intent, **extra_meta},
                )
                messages = [
                    ChatMessageView(role=m.role, content=m.content, metadata=m.metadata)
                    for m in db.list_chat_messages(session_id)
                ]
                return AgentExperienceResponse(
                    session_id=session_id, answer=answer, citations=citations,
                    documents=[_document_view(d) for d in documents],
                    messages=messages, hook_metadata=merged_metadata, intent=intent,
                )
            # --- Explicit mode (backwards compat) ---
            # Keep existing if/elif chain below, updating each branch to
            # return AgentExperienceResponse with intent= set.
```

Then in each existing explicit-mode `return AgentExperienceResponse(...)` call, add `intent=`:
- `search_tool` branch → `intent="search"`
- `hybrid_search` branch → `intent="search"`
- `chat_loop` branch → `intent="chat"`
- `search_agent` branch → `intent="search"`
- `tool_agent` branch → `intent="tool"`
- The final default `answer_with_retrieval` path (chat_once) → `intent="chat"`

For example, the `search_tool` branch currently ends with:
```python
return AgentExperienceResponse(
    session_id=session_id, answer=answer, citations=[doc.citation for doc in documents],
    documents=[_document_view(doc) for doc in documents],
    messages=messages, hook_metadata=hook_metadata,
)
```
Becomes:
```python
return AgentExperienceResponse(
    session_id=session_id, answer=answer, citations=[doc.citation for doc in documents],
    documents=[_document_view(doc) for doc in documents],
    messages=messages, hook_metadata=hook_metadata, intent="search",
)
```
Apply the same pattern to every other explicit-mode return.

Also update the catch-all exception handler to replace "Agent search failed":
```python
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Agent dispatch error: %s", exc)
            detail = str(exc) if str(exc) else "Unexpected error during agent dispatch"
            raise HTTPException(status_code=502, detail=detail) from exc
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/unit/servers/web/test_web_experience_app.py -v
```

Expected: all tests pass including the 3 new ones.

- [ ] **Step 6: Commit**

```bash
git add src/internal/servers/web/app.py
git commit -m "feat: add _run_auto_routed with three-tier dispatch, wire into run_agent"
```

---

## Task 5: Execution fallback tests

**Files:**
- Create: `tests/unit/test_execution_fallbacks.py`

- [ ] **Step 1: Write all fallback tests**

```python
# tests/unit/test_execution_fallbacks.py
"""Tests for mid-execution fallbacks in _run_auto_routed."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from src.context.models import AnswerGenerationResult, SearchContextBundle, PromptBundle, ContextDocument
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app
from src.agents.base import AgentLoopOutput


def _make_answer_result(answer: str = "ok") -> AnswerGenerationResult:
    return AnswerGenerationResult(
        answer=answer, citations=[],
        context=SearchContextBundle(query="q", documents=[]),
        prompt=PromptBundle(system="", user="", messages=[]),
    )


def _make_loop_output(final_answer: str | None = "tool answer") -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[], response_ids=[], response_mask=[],
        num_turns=1, final_answer=final_answer,
    )


# --- Intent model fallbacks ---

def test_tool_loop_raises_reroutes_to_tier2(monkeypatch, tmp_path):
    """ToolAgentLoop.run raises → falls back to answer_with_retrieval via Tier 2."""
    monkeypatch.setattr(
        "src.agents.tool_calling.ToolAgentLoop.run",
        AsyncMock(side_effect=RuntimeError("OOM")),
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.classify_is_search_flow",
        lambda q, llm: False,
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        AsyncMock(return_value=_make_answer_result("tier2 answer")),
    )
    # Need an llm configured — monkeypatch the app's llm
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    # Simulate manager being set (so Tier 1 is attempted)
    with TestClient(app) as client:
        # Inject fake manager into app state
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        # Also inject an llm (patch at module level)
        monkeypatch.setattr("src.internal.servers.web.app.llm", MagicMock(), raising=False)
        response = client.post("/api/agent", json={"query": "what is FAISS"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "chat"
    assert "intent_fallback" in str(data)


def test_tool_loop_empty_output_reroutes(monkeypatch, tmp_path):
    """ToolAgentLoop returns final_answer=None → falls back."""
    monkeypatch.setattr(
        "src.agents.tool_calling.ToolAgentLoop.run",
        AsyncMock(return_value=_make_loop_output(final_answer=None)),
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.classify_is_search_flow",
        lambda q, llm: False,
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        AsyncMock(return_value=_make_answer_result("fallback answer")),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        response = client.post("/api/agent", json={"query": "explain FAISS"})
    assert response.status_code == 200
    assert response.json()["intent"] == "chat"


# --- Search fallbacks ---

def test_hybrid_search_fails_falls_back_to_rag_without_context(monkeypatch, tmp_path):
    """_run_hybrid_search raises → answer_with_retrieval called with top_k=0."""
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_hybrid_search",
        AsyncMock(side_effect=ConnectionError("retrieval down")),
    )
    rag_call_kwargs: dict = {}
    async def fake_rag(q, *, llm, chat_history, search_url, top_k, filters):
        rag_call_kwargs["top_k"] = top_k
        return _make_answer_result("rag fallback")
    monkeypatch.setattr("src.internal.servers.web.app.answer_with_retrieval", fake_rag)
    monkeypatch.setattr(
        "src.internal.servers.web.app.classify_is_search_flow",
        lambda q, llm: True,  # classified as search → hybrid fails → falls to chat
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "find onboarding doc"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "chat"
    assert rag_call_kwargs["top_k"] == 0  # called with no context


def test_hybrid_search_and_rag_both_fail_returns_502(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_hybrid_search",
        AsyncMock(side_effect=ConnectionError("down")),
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        AsyncMock(side_effect=RuntimeError("llm down")),
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_direct_search",
        AsyncMock(side_effect=ConnectionError("still down")),
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.classify_is_search_flow",
        lambda q, llm: True,
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "find doc"})
    assert response.status_code == 502
    assert "retrieval also unavailable" in response.json()["detail"].lower()


# --- RAG fallbacks ---

def test_answer_with_retrieval_fails_falls_back_to_raw_docs(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.internal.servers.web.app.classify_is_search_flow",
        lambda q, llm: False,
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        AsyncMock(side_effect=RuntimeError("llm down")),
    )
    raw_doc = ContextDocument(id="D1", title="Doc", content="content", url=None, score=0.9, metadata={})
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_direct_search",
        AsyncMock(return_value=[raw_doc, raw_doc, raw_doc]),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "explain FAISS"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "search"
    assert len(data["documents"]) == 3
    assert "rag_fallback" in str(data)


def test_answer_with_retrieval_and_search_both_fail_returns_502(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.internal.servers.web.app.classify_is_search_flow",
        lambda q, llm: False,
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        AsyncMock(side_effect=RuntimeError("llm down")),
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_direct_search",
        AsyncMock(side_effect=ConnectionError("retrieval down")),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "explain FAISS"})
    assert response.status_code == 502
    assert "retrieval also unavailable" in response.json()["detail"].lower()
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/unit/test_execution_fallbacks.py -v
```

Expected: All pass. If any fail, the corresponding `_run_auto_routed` branch is missing or incorrect — fix `app.py` until they pass.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_execution_fallbacks.py
git commit -m "test: add execution fallback tests for intent model, search, and RAG"
```

---

## Task 6: Fix error handling + explicit-mode 400s

**Files:**
- Modify: `src/internal/servers/web/app.py` (error messages in explicit mode branches)
- Test: `tests/unit/servers/web/test_web_experience_app.py` (append)

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/unit/servers/web/test_web_experience_app.py

def test_agent_no_llm_no_model_returns_400(tmp_path):
    """App with no LLM and no local model → chat query → 400."""
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "explain FAISS"})
    assert response.status_code == 400
    assert "no llm" in response.json()["detail"].lower()


def test_agent_tool_mode_without_model_returns_clear_400(tmp_path):
    """Explicit mode=tool_agent without local model → 400 with 'local model' in detail."""
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "run tool", "mode": "tool_agent"})
    assert response.status_code == 400
    assert "local model" in response.json()["detail"].lower()


def test_agent_other_exception_returns_502_with_message(monkeypatch, tmp_path):
    """Unexpected exception → 502 with the exception message, not 'Agent search failed'."""
    async def explode(*args, **kwargs):
        raise ValueError("bad input format")
    monkeypatch.setattr("src.internal.servers.web.app.answer_with_retrieval", explode)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "explain FAISS", "mode": "chat_once"})
    assert response.status_code == 502
    assert "bad input format" in response.json()["detail"]
    assert "Agent search failed" not in response.json()["detail"]
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/unit/servers/web/test_web_experience_app.py -k "400 or 502_with_message" -v
```

- [ ] **Step 3: Update error handling in explicit mode branches of `run_agent`**

In the `search_agent` and `tool_agent` branches (around lines 576–700 of app.py), the existing 400 errors already have messages. Verify they say "local model":

```python
# search_agent branch (already exists, verify message):
raise HTTPException(
    status_code=400,
    detail=(
        "search_agent mode is not configured. "
        "Set SEARCH_AGENT_MODEL in .env and restart the server."
    ),
)

# tool_agent branch — update to say "local model":
raise HTTPException(
    status_code=400,
    detail=(
        "tool_agent mode requires a local model. "
        "Set SEARCH_AGENT_MODEL or SEARCH_AGENT_SERVER_URL in .env and restart."
    ),
)
```

Update the catch-all at the end of `run_agent` (replace the existing one):
```python
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Agent dispatch error: %s", exc)
            detail = str(exc) if str(exc).strip() else "Unexpected agent dispatch error"
            raise HTTPException(status_code=502, detail=detail) from exc
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/servers/web/test_web_experience_app.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/internal/servers/web/app.py tests/unit/servers/web/test_web_experience_app.py
git commit -m "fix: replace 'Agent search failed' with specific error messages"
```

---

## Task 7: Frontend types + api.test.ts

**Files:**
- Modify: `web/src/types.ts`
- Create: `web/src/__tests__/api.test.ts`

- [ ] **Step 1: Write failing api tests**

```typescript
// web/src/__tests__/api.test.ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import { runAgent, streamAgent } from "../api";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

beforeEach(() => { mockFetch.mockReset(); });

describe("runAgent", () => {
  it("passes intent field through from response", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        session_id: "s1", answer: "hello", citations: [],
        documents: [], messages: [], intent: "search",
      }),
    });
    const result = await runAgent({ query: "find docs" });
    expect(result.intent).toBe("search");
  });
});

describe("streamAgent", () => {
  it("yields error event when response is not ok", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 502,
      body: null,
    });
    const gen = streamAgent({ query: "q" });
    await expect(gen.next()).rejects.toThrow("502");
  });
});
```

- [ ] **Step 2: Run tests — expect type error (intent missing from type)**

```bash
cd web && npm run typecheck
```

Expected: TypeScript error about `intent` not on `AgentExperienceResponse`.

- [ ] **Step 3: Update `web/src/types.ts`**

```typescript
// In AgentExperienceResponse — add intent field:
export interface AgentExperienceResponse {
  session_id: string;
  answer: string;
  citations: string[];
  documents: SourceDocumentView[];
  messages: ChatMessageView[];
  intent?: "search" | "chat" | "tool";
}

// Update AgentMode — remove search_agent and tool_agent:
export type AgentMode = "search_tool" | "hybrid_search" | "chat_once" | "chat_loop";
```

- [ ] **Step 4: Run typecheck + tests**

```bash
cd web && npm run typecheck && npm run test -- src/__tests__/api.test.ts
```

Expected: no type errors, tests pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/types.ts web/src/__tests__/api.test.ts
git commit -m "feat: add intent to AgentExperienceResponse type, trim AgentMode"
```

---

## Task 8: Remove mode dropdown from SearchComposer

**Files:**
- Modify: `web/src/components/SearchComposer.tsx`
- Modify: `web/src/components/__tests__/SearchComposer.test.tsx`

- [ ] **Step 1: Update tests first (TDD)**

Replace the content of `SearchComposer.test.tsx`:

```typescript
// web/src/components/__tests__/SearchComposer.test.tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SearchComposer } from "../SearchComposer";

const defaultProps = {
  query: "",
  searchUrl: "http://localhost:8001",
  topK: 5,
  sourceProvider: "retrieval" as const,
  isLoading: false,
  onQueryChange: vi.fn(),
  onSearchUrlChange: vi.fn(),
  onTopKChange: vi.fn(),
  onSourceProviderChange: vi.fn(),
  onSubmit: vi.fn(),
};

describe("SearchComposer", () => {
  it("renders a textarea and submit button", () => {
    render(<SearchComposer {...defaultProps} />);
    expect(screen.getByRole("textbox", { name: /question/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /search/i })).toBeInTheDocument();
  });

  it("does NOT render a mode dropdown", () => {
    render(<SearchComposer {...defaultProps} />);
    expect(screen.queryByLabelText(/entry point/i)).not.toBeInTheDocument();
  });

  it("renders retrieval URL and topK fields", () => {
    render(<SearchComposer {...defaultProps} />);
    expect(screen.getByDisplayValue("http://localhost:8001")).toBeInTheDocument();
    expect(screen.getByDisplayValue("5")).toBeInTheDocument();
  });

  it("disables submit when query is empty", () => {
    render(<SearchComposer {...defaultProps} query="" />);
    expect(screen.getByRole("button", { name: /search/i })).toBeDisabled();
  });

  it("enables submit when query has content", () => {
    render(<SearchComposer {...defaultProps} query="What is FAISS?" />);
    expect(screen.getByRole("button", { name: /search/i })).not.toBeDisabled();
  });

  it("disables submit while loading", () => {
    render(<SearchComposer {...defaultProps} query="hello" isLoading={true} />);
    expect(screen.getByRole("button")).toBeDisabled();
  });

  it("calls onSubmit when form is submitted", async () => {
    const onSubmit = vi.fn();
    render(<SearchComposer {...defaultProps} query="test" onSubmit={onSubmit} />);
    await userEvent.click(screen.getByRole("button"));
    expect(onSubmit).toHaveBeenCalledOnce();
  });

  it("submits on Cmd+Enter", async () => {
    const onSubmit = vi.fn();
    render(<SearchComposer {...defaultProps} query="hello" onSubmit={onSubmit} />);
    const textarea = screen.getByRole("textbox", { name: /question/i });
    await userEvent.click(textarea);
    await userEvent.keyboard("{Meta>}{Enter}{/Meta}");
    expect(onSubmit).toHaveBeenCalledOnce();
  });

  it("calls onQueryChange when user types", async () => {
    const onQueryChange = vi.fn();
    render(<SearchComposer {...defaultProps} onQueryChange={onQueryChange} />);
    await userEvent.type(screen.getByRole("textbox", { name: /question/i }), "hello");
    expect(onQueryChange).toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run tests — expect FAIL (mode props still required)**

```bash
cd web && npm run test -- SearchComposer
```

Expected: TypeScript compile error about `mode` and `onModeChange` being required.

- [ ] **Step 3: Update `SearchComposer.tsx`**

```tsx
// web/src/components/SearchComposer.tsx
import { memo } from "react";
import type { FormEvent } from "react";
import { Loader2, Search } from "lucide-react";
import type { SearchSourceProvider } from "../types";

const SOURCE_OPTIONS: Array<{
  value: SearchSourceProvider;
  label: string;
  disabled?: boolean;
}> = [
  { value: "retrieval", label: "Local Retrieval" },
  { value: "google", label: "Google PSE", disabled: true },
  { value: "serpapi", label: "SerpAPI" },
  { value: "browser", label: "Browser Retrieval" },
  { value: "all", label: "All Active Sources" },
];

interface SearchComposerProps {
  query: string;
  searchUrl: string;
  topK: number;
  sourceProvider: SearchSourceProvider;
  isLoading: boolean;
  onQueryChange: (value: string) => void;
  onSearchUrlChange: (value: string) => void;
  onTopKChange: (value: number) => void;
  onSourceProviderChange: (value: SearchSourceProvider) => void;
  onSubmit: (event?: FormEvent) => void;
}

export const SearchComposer = memo(function SearchComposer({
  query,
  searchUrl,
  topK,
  sourceProvider,
  isLoading,
  onQueryChange,
  onSearchUrlChange,
  onTopKChange,
  onSourceProviderChange,
  onSubmit,
}: SearchComposerProps) {
  return (
    <form className="composer" onSubmit={onSubmit}>
      <textarea
        aria-label="Question"
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
            e.preventDefault();
            onSubmit();
          }
        }}
        placeholder="Ask about your indexed docs, web results, or retrieval server output"
        rows={4}
      />
      <div className="composer-controls">
        <label>
          Source
          <select
            value={sourceProvider}
            onChange={(e) => onSourceProviderChange(e.currentTarget.value as SearchSourceProvider)}
          >
            {SOURCE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value} disabled={opt.disabled}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>

        <label className="url-field">
          Retrieval URL
          <input value={searchUrl} onChange={(e) => onSearchUrlChange(e.target.value)} />
        </label>

        <label>
          Top K
          <input
            min={1} max={20} type="number" value={topK}
            onChange={(e) => onTopKChange(e.currentTarget.valueAsNumber)}
          />
        </label>

        <button type="submit" disabled={isLoading || !query.trim()}>
          {isLoading ? <Loader2 className="spin" size={18} /> : <Search size={18} />}
          <span>{isLoading ? "Searching" : "Search"}</span>
        </button>
      </div>
    </form>
  );
});
```

- [ ] **Step 4: Run tests**

```bash
cd web && npm run test -- SearchComposer && npm run typecheck
```

Expected: all 9 SearchComposer tests pass, no type errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/SearchComposer.tsx web/src/components/__tests__/SearchComposer.test.tsx
git commit -m "feat: remove mode dropdown from SearchComposer, intent is now auto-detected"
```

---

## Task 9: AnswerPanel intent badge

**Files:**
- Modify: `web/src/components/AnswerPanel.tsx`
- Modify: `web/src/components/__tests__/AnswerPanel.test.tsx`

- [ ] **Step 1: Update tests**

```typescript
// web/src/components/__tests__/AnswerPanel.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AnswerPanel } from "../AnswerPanel";

describe("AnswerPanel", () => {
  it("renders empty state when answer is empty", () => {
    render(<AnswerPanel answer="" citations={[]} />);
    expect(screen.getByText(/results will appear here/i)).toBeInTheDocument();
  });

  it("renders the answer text", () => {
    render(<AnswerPanel answer="FAISS is a vector library." citations={[]} />);
    expect(screen.getByText(/FAISS is a vector library/)).toBeInTheDocument();
  });

  it("renders citation chips when present", () => {
    render(<AnswerPanel answer="See [D1] for details." citations={["[D1]"]} />);
    expect(screen.getByText("[D1]")).toBeInTheDocument();
  });

  it("does not render citation row when citations are empty", () => {
    render(<AnswerPanel answer="Some answer." citations={[]} />);
    expect(screen.queryByLabelText(/citations/i)).not.toBeInTheDocument();
  });

  it("renders 'Searched · 5 sources' badge when intent is search", () => {
    render(<AnswerPanel answer="results" citations={[]} intent="search" documentCount={5} />);
    expect(screen.getByText(/searched · 5 sources/i)).toBeInTheDocument();
  });

  it("renders 'Answered · 2 citations' badge when intent is chat", () => {
    render(<AnswerPanel answer="answer" citations={["[D1]", "[D2]"]} intent="chat" />);
    expect(screen.getByText(/answered · 2 citations/i)).toBeInTheDocument();
  });

  it("renders 'Used tools · 3 calls' badge when intent is tool", () => {
    render(<AnswerPanel answer="tool output" citations={[]} intent="tool" toolCallCount={3} />);
    expect(screen.getByText(/used tools · 3 calls/i)).toBeInTheDocument();
  });

  it("renders no badge when intent is undefined", () => {
    render(<AnswerPanel answer="answer" citations={[]} />);
    expect(document.querySelector(".intent-badge")).not.toBeInTheDocument();
  });

  it("renders no badge when answer is empty even if intent is set", () => {
    render(<AnswerPanel answer="" citations={[]} intent="chat" />);
    expect(document.querySelector(".intent-badge")).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run tests — expect FAIL (badge not implemented)**

```bash
cd web && npm run test -- AnswerPanel
```

Expected: 5 badge tests fail.

- [ ] **Step 3: Update `AnswerPanel.tsx`**

```tsx
// web/src/components/AnswerPanel.tsx
import { memo, useMemo } from "react";

interface AnswerPanelProps {
  answer: string;
  citations: string[];
  intent?: "search" | "chat" | "tool";
  documentCount?: number;
  toolCallCount?: number;
}

function IntentBadge({ intent, citations, documentCount, toolCallCount }: {
  intent: "search" | "chat" | "tool";
  citations: string[];
  documentCount?: number;
  toolCallCount?: number;
}) {
  if (intent === "search") {
    const n = documentCount ?? 0;
    return <span className="intent-badge">Searched · {n} {n === 1 ? "source" : "sources"}</span>;
  }
  if (intent === "chat") {
    const n = citations.length;
    return <span className="intent-badge">Answered · {n} {n === 1 ? "citation" : "citations"}</span>;
  }
  const n = toolCallCount ?? 0;
  return <span className="intent-badge">Used tools · {n} {n === 1 ? "call" : "calls"}</span>;
}

export const AnswerPanel = memo(function AnswerPanel({
  answer,
  citations,
  intent,
  documentCount,
  toolCallCount,
}: AnswerPanelProps) {
  const paragraphs = useMemo(() => answer.split(/\n\n+/).filter(Boolean), [answer]);

  if (!answer) {
    return (
      <div className="empty-state">
        Results will appear here once the agent retrieves context.
      </div>
    );
  }

  return (
    <article className="answer-panel">
      {intent && (
        <IntentBadge
          intent={intent}
          citations={citations}
          documentCount={documentCount}
          toolCallCount={toolCallCount}
        />
      )}
      {paragraphs.map((para, i) => (
        <p key={i}>{para}</p>
      ))}
      {citations.length > 0 && (
        <div className="citation-row" aria-label="Citations">
          {citations.map((citation) => (
            <span key={citation}>{citation}</span>
          ))}
        </div>
      )}
    </article>
  );
});
```

- [ ] **Step 4: Run tests**

```bash
cd web && npm run test -- AnswerPanel && npm run typecheck
```

Expected: all 9 AnswerPanel tests pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/AnswerPanel.tsx web/src/components/__tests__/AnswerPanel.test.tsx
git commit -m "feat: add intent badge to AnswerPanel (Searched/Answered/Used tools)"
```

---

## Task 10: App.tsx adaptive layout + App.test.tsx

**Files:**
- Modify: `web/src/App.tsx`
- Create: `web/src/components/__tests__/App.test.tsx`

- [ ] **Step 1: Write App tests**

```typescript
// web/src/components/__tests__/App.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { App } from "../../App";

vi.mock("../../api", () => ({
  createSession: vi.fn().mockResolvedValue({ id: "s1", messages: [], title: null, user_id: null }),
  runAgent: vi.fn(),
  streamAgent: vi.fn(),
  getAdminSummary: vi.fn().mockRejectedValue(new Error("no admin")),
  getAnalyticsByLLM: vi.fn().mockRejectedValue(new Error()),
  getAnalyticsByPersona: vi.fn().mockRejectedValue(new Error()),
  getAnalyticsByFlow: vi.fn().mockRejectedValue(new Error()),
}));

import * as api from "../../api";

const mockRunAgent = api.runAgent as ReturnType<typeof vi.fn>;

const baseResponse = {
  session_id: "s1", answer: "The answer", citations: ["[D1]"],
  documents: [{ id: "D1", citation: "[D1]", title: "Doc", content: "c", url: null, score: 0.9, metadata: {} }],
  messages: [{ role: "user", content: "q", metadata: {} }, { role: "assistant", content: "The answer", metadata: {} }],
};

beforeEach(() => { vi.clearAllMocks(); });

async function submitQuery(query = "explain FAISS") {
  const textarea = screen.getByRole("textbox", { name: /question/i });
  await userEvent.clear(textarea);
  await userEvent.type(textarea, query);
  await userEvent.click(screen.getByRole("button", { name: /search/i }));
}

describe("App adaptive layout", () => {
  it("adds intent-search class to results layout on search response", async () => {
    mockRunAgent.mockResolvedValue({ ...baseResponse, intent: "search" });
    render(<App />);
    await submitQuery("find the onboarding doc");
    await waitFor(() => {
      const layout = document.querySelector(".results-layout");
      expect(layout?.classList).toContain("intent-search");
    });
  });

  it("adds intent-chat class to results layout on chat response", async () => {
    mockRunAgent.mockResolvedValue({ ...baseResponse, intent: "chat" });
    render(<App />);
    await submitQuery("explain FAISS");
    await waitFor(() => {
      const layout = document.querySelector(".results-layout");
      expect(layout?.classList).toContain("intent-chat");
    });
  });

  it("adds intent-tool class to results layout on tool response", async () => {
    mockRunAgent.mockResolvedValue({ ...baseResponse, intent: "tool", documents: [] });
    render(<App />);
    await submitQuery("run API tool");
    await waitFor(() => {
      const layout = document.querySelector(".results-layout");
      expect(layout?.classList).toContain("intent-tool");
    });
  });

  it("shows error banner on API failure", async () => {
    mockRunAgent.mockRejectedValue(new Error("No LLM configured"));
    render(<App />);
    await submitQuery("explain FAISS");
    await waitFor(() => {
      expect(screen.getByText(/no llm configured/i)).toBeInTheDocument();
    });
  });

  it("clears error when new session is created", async () => {
    mockRunAgent.mockRejectedValue(new Error("failed"));
    render(<App />);
    await submitQuery("q");
    await waitFor(() => expect(screen.getByText(/failed/i)).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: /new/i }));
    await waitFor(() => expect(screen.queryByText(/failed/i)).not.toBeInTheDocument());
  });
});
```

- [ ] **Step 2: Run tests — expect FAIL (intent class not applied)**

```bash
cd web && npm run test -- App.test
```

Expected: intent class tests fail.

- [ ] **Step 3: Update `App.tsx`**

Make these changes to `App.tsx`:
1. Remove `mode` and `setMode` state, `isChatMode`, `isSearchMode`, `STREAMING_MODES`
2. Add `intent` state: `const [intent, setIntent] = useState<"search" | "chat" | "tool" | undefined>(undefined)`
3. In `handleSubmit`, remove `mode` from `agentRequest`, set `intent` from response
4. Pass `intent`, `documentCount` to `AnswerPanel`
5. Add intent class to `results-layout` div
6. Remove `mode`, `onModeChange` from `SearchComposer` props

Key diffs:

```tsx
// Remove:
const [mode, setMode] = useState<AgentMode>("chat_once");
const isChatMode = mode === "chat_once" || mode === "chat_loop";
const isSearchMode = mode === "search_tool" || mode === "hybrid_search";
const STREAMING_MODES: AgentMode[] = ["search_agent", "tool_agent", "chat_loop"];

// Add:
const [intent, setIntent] = useState<"search" | "chat" | "tool" | undefined>(undefined);

// In handleSubmit, update agentRequest:
const agentRequest: AgentExperienceRequest = {
  query: normalizedQuery,
  session_id: activeSessionId,
  search_url: searchUrl,
  top_k: topK,
  source_provider: sourceProvider,
  // no mode field — auto-detected by backend
};

// After setting answer:
setIntent(response.intent);

// In handleNewSession, add:
setIntent(undefined);

// results-layout div:
<div className={`results-layout${intent ? ` intent-${intent}` : ""}`}>

// AnswerPanel usage:
<AnswerPanel
  answer={streamingAnswer || answer}
  citations={citations}
  intent={intent}
  documentCount={documents.length}
/>

// SearchComposer — remove mode and onModeChange props:
<SearchComposer
  query={query}
  searchUrl={searchUrl}
  topK={topK}
  sourceProvider={sourceProvider}
  isLoading={isLoading}
  onQueryChange={setQuery}
  onSearchUrlChange={setSearchUrl}
  onTopKChange={handleTopKChange}
  onSourceProviderChange={handleSourceProviderChange}
  onSubmit={handleSubmit}
/>
```

- [ ] **Step 4: Run all frontend tests + typecheck**

```bash
cd web && npm run test && npm run typecheck
```

Expected: all tests pass, no type errors.

- [ ] **Step 5: Run full backend test suite**

```bash
cd /Users/linghuang/Git/Agentic-Search && pytest tests/unit/ -v --tb=short
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add web/src/App.tsx web/src/components/__tests__/App.test.tsx
git commit -m "feat: App adapts layout by intent, removes mode selector"
```

---

## Final verification

- [ ] **Run complete test suite**

```bash
pytest tests/unit/ -v --tb=short
cd web && npm run test && npm run typecheck
```

Expected: all pass.

- [ ] **Smoke test the stack**

```bash
# Terminal 1
python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl

# Terminal 2
PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860

# Terminal 3
cd web && npm run dev
```

Open `http://127.0.0.1:5173`. Verify:
- No mode dropdown visible
- Typing "explain FAISS" → answer panel prominent, intent badge says "Answered · N citations"
- Typing "procurement process" → source grid prominent, intent badge says "Searched · N sources"
- Error messages are specific (stop the retrieval server, verify "Cannot reach retrieval server" appears)

- [ ] **Open PR**

```bash
git push -u origin feat/intent-routed-search-chat
gh pr create --title "feat: intent-routed search & chat with tiered fallback" --body "$(cat <<'EOF'
## Summary
- Replaces manual mode dropdown with automatic intent routing
- ToolAgentLoop acts as universal router (search_routing_tool + rag_routing_tool + MCP tools)
- Three-tier fallback: ToolAgentLoop → LLM classifier → rule-based keyword matcher
- Execution-level fallbacks: intent model crash, search failure, RAG failure all degrade gracefully
- Frontend adapts layout class (intent-search/chat/tool) and shows intent badge
- "Agent search failed" replaced with specific, actionable error messages

## Test plan
- [ ] `pytest tests/unit/` — all pass
- [ ] `cd web && npm run test && npm run typecheck` — all pass
- [ ] Smoke test: type a search query, verify source grid is prominent
- [ ] Smoke test: type a chat query, verify answer panel is prominent
- [ ] Smoke test: stop retrieval server, verify error message contains URL

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
