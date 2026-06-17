# Tool Call Trace Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface `ToolAgentLoop` execution traces as a `ToolCallTracePanel` component in the frontend, showing each tool call's name, status, arguments, result summary, and latency.

**Architecture:** Three layers: (1) Backend — add `arguments` field to `ToolExecutionResult`, extend `_run_auto_routed` trace parsing to produce `ToolCallView` list, add to `AgentExperienceResponse`; (2) Types — `ToolCallTraceView` in `types.ts`, `tool_calls` on `SSEDoneEvent` and `AgentExperienceResponse`; (3) Frontend — new `ToolCallTracePanel` component, `toolCalls` state in `App.tsx`.

**Tech Stack:** Python/Pydantic (backend), React 19 + TypeScript + Vitest (frontend). Worktree: `.worktrees/feat-tool-trace`, branch `feat/tool-trace`.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| **Modify** | `src/agents/state.py` | Add `arguments: dict` field to `ToolExecutionResult` |
| **Modify** | `src/agents/tool_calling.py` | Pass `tool_call.parsed_arguments()` into `ToolExecutionResult` |
| **Modify** | `src/internal/servers/web/app.py` | Add `ToolCallView` Pydantic model; extend trace parsing; add `tool_calls` to `AgentExperienceResponse`; add `tool_calls` to SSE `done` event |
| **Modify** | `web/src/types.ts` | Add `ToolCallTraceView`; add `tool_calls?` to `AgentExperienceResponse` and `SSEDoneEvent` |
| **Create** | `web/src/components/ToolCallTracePanel.tsx` | New panel component |
| **Modify** | `web/src/App.tsx` | Add `toolCalls` state; handle `tool_calls` from SSE done event; render panel when `intent === "tool"` |
| **Modify** | `web/src/styles.css` | Add `.tool-trace-*` styles |
| **Create** | `tests/unit/servers/web/test_tool_trace.py` | Backend unit tests for trace parsing |
| **Create** | `web/src/components/__tests__/ToolCallTracePanel.test.tsx` | Frontend component tests |

---

## Task 1: Backend — `arguments` field + `ToolCallView` model + trace parsing

**Files:**
- Modify: `src/agents/state.py`
- Modify: `src/agents/tool_calling.py`
- Modify: `src/internal/servers/web/app.py`
- Create: `tests/unit/servers/web/test_tool_trace.py`

### Step 1: Write failing backend tests

Create `tests/unit/servers/web/test_tool_trace.py`:

```python
"""Tests for ToolCallView trace parsing in _run_auto_routed."""
from __future__ import annotations

import json
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from src.internal.servers.web.app import SearchExperienceSettings, create_web_app
from src.agents.base import AgentLoopOutput


def _make_output(action_trace: str) -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        final_answer="done",
        action_trace=action_trace,
    )


def _trace_line(tool_name, status, result, arguments=None, execution_time=0.123, error_message=None):
    return json.dumps({
        "tool_name": tool_name,
        "status": status,
        "result": result,
        "arguments": arguments or {},
        "performance": {"execution_time": execution_time},
        "error_message": error_message,
    })


def test_tool_calls_populated_from_action_trace(monkeypatch, tmp_path):
    """AgentExperienceResponse.tool_calls has one entry per trace line."""
    trace = "\n".join([
        _trace_line("search_routing_tool", "TaskStatus.COMPLETED", json.dumps([{"title": "t", "content": "c", "url": None}]), {"query": "q"}),
        _trace_line("some_other_tool", "TaskStatus.COMPLETED", "plain result", {"x": 1}, execution_time=0.05),
    ])
    monkeypatch.setattr(
        "src.agents.tool_calling.ToolAgentLoop.run",
        AsyncMock(return_value=_make_output(trace)),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        response = client.post("/api/agent", json={"query": "test"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["tool_calls"]) == 2
    assert data["tool_calls"][0]["tool_name"] == "search_routing_tool"
    assert data["tool_calls"][1]["tool_name"] == "some_other_tool"


def test_latency_computed_from_execution_time(monkeypatch, tmp_path):
    """latency_ms is int(execution_time * 1000)."""
    trace = _trace_line("my_tool", "TaskStatus.COMPLETED", "ok", execution_time=0.456)
    monkeypatch.setattr(
        "src.agents.tool_calling.ToolAgentLoop.run",
        AsyncMock(return_value=_make_output(trace)),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        response = client.post("/api/agent", json={"query": "test"})
    assert response.status_code == 200
    assert response.json()["tool_calls"][0]["latency_ms"] == 456


def test_list_result_becomes_n_items(monkeypatch, tmp_path):
    """A list result is summarised as 'N items'."""
    trace = _trace_line("search_routing_tool", "TaskStatus.COMPLETED",
                        json.dumps([{"title": "a"}, {"title": "b"}]))
    monkeypatch.setattr(
        "src.agents.tool_calling.ToolAgentLoop.run",
        AsyncMock(return_value=_make_output(trace)),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        response = client.post("/api/agent", json={"query": "test"})
    assert response.status_code == 200
    assert response.json()["tool_calls"][0]["result_summary"] == "2 items"


def test_string_result_truncated_to_200(monkeypatch, tmp_path):
    """A long string result is truncated to 200 chars."""
    long_result = "x" * 300
    trace = _trace_line("my_tool", "TaskStatus.COMPLETED", long_result)
    monkeypatch.setattr(
        "src.agents.tool_calling.ToolAgentLoop.run",
        AsyncMock(return_value=_make_output(trace)),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        response = client.post("/api/agent", json={"query": "test"})
    assert response.status_code == 200
    assert len(response.json()["tool_calls"][0]["result_summary"]) == 200


def test_failed_tool_call_error_message(monkeypatch, tmp_path):
    """error_message is mapped to ToolCallView.error for failed calls."""
    trace = _trace_line("bad_tool", "TaskStatus.FAILED", None, error_message="timeout")
    monkeypatch.setattr(
        "src.agents.tool_calling.ToolAgentLoop.run",
        AsyncMock(return_value=_make_output(trace)),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        response = client.post("/api/agent", json={"query": "test"})
    assert response.status_code == 200
    tc = response.json()["tool_calls"][0]
    assert tc["status"] == "failed"
    assert tc["error"] == "timeout"


def test_no_tool_calls_on_chat_path(monkeypatch, tmp_path):
    """Non-tool paths return tool_calls=[]."""
    from src.context.models import AnswerGenerationResult, SearchContextBundle, PromptBundle
    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        AsyncMock(return_value=AnswerGenerationResult(
            answer="hi",
            citations=[],
            context=SearchContextBundle(query="q", documents=[]),
            prompt=PromptBundle(system="", user="", messages=[]),
        )),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "explain FAISS"})
    assert response.status_code == 200
    assert response.json()["tool_calls"] == []
```

- [ ] **Step 2: Run failing tests**

```bash
cd /Users/linghuang/Git/Agentic-Search/.worktrees/feat-tool-trace
pytest tests/unit/servers/web/test_tool_trace.py -v 2>&1 | tail -20
```

Expected: tests FAIL because `tool_calls` key is absent from the response.

- [ ] **Step 3: Add `arguments` field to `ToolExecutionResult`**

In `src/agents/state.py`, find `ToolExecutionResult` (around line 150). Add `arguments` field:

```python
@dataclass(slots=True)
class ToolExecutionResult:
    tool_name: str
    status: TaskStatus
    result: Any
    arguments: dict[str, Any] = field(default_factory=dict)   # add this
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    error_code: str | None = None
    error_message: str | None = None
    optimization_suggestions: list[str] = field(default_factory=list)
    retry_count: int = 0
```

- [ ] **Step 4: Pass `parsed_arguments()` in `_call_tool`**

In `src/agents/tool_calling.py`, find the `return ToolExecutionResult(...)` inside `_call_tool` (around line 188). Add `arguments=tool_call.parsed_arguments()`:

```python
return ToolExecutionResult(
    tool_name=tool_call.name,
    status=status,
    result=result,
    arguments=tool_call.parsed_arguments(),   # add this line
    performance=PerformanceMetrics(
        execution_time=elapsed,
        success_rate=1.0 if status is TaskStatus.COMPLETED else 0.0,
    ),
    error_code=error_code,
    error_message=error_message,
)
```

- [ ] **Step 5: Add `ToolCallView` Pydantic model to `app.py`**

In `src/internal/servers/web/app.py`, find `class AgentExperienceResponse(BaseModel):` and add `ToolCallView` just before it:

```python
class ToolCallView(BaseModel):
    tool_name: str
    status: str           # "completed" | "failed"
    arguments: dict[str, object]
    result_summary: str   # first 200 chars, or "N items" for lists
    latency_ms: int
    error: str | None = None
```

- [ ] **Step 6: Add `tool_calls` field to `AgentExperienceResponse`**

In `AgentExperienceResponse`, add:
```python
tool_calls: list[ToolCallView] = Field(default_factory=list)
```

- [ ] **Step 7: Extend `action_trace` parsing in `_run_auto_routed`**

In `_run_auto_routed`, find the `if output.action_trace:` block (around line 323). Replace the existing loop with an extended version that collects `ToolCallView` objects AND still extracts documents from `search_routing_tool`:

```python
tool_calls: list[ToolCallView] = []
documents = []
if output.action_trace:
    for line in output.action_trace.split("\n"):
        if not line.strip():
            continue
        try:
            rec = _json.loads(line)
            tool_name = rec.get("tool_name", "")
            perf = rec.get("performance", {})
            latency_ms = int(perf.get("execution_time", 0.0) * 1000)
            status_raw = str(rec.get("status", "failed")).lower()
            is_completed = "completed" in status_raw
            result = rec.get("result")

            if isinstance(result, list):
                result_summary = f"{len(result)} items"
            elif result is not None:
                result_summary = str(result)[:200]
            else:
                result_summary = ""

            tool_calls.append(ToolCallView(
                tool_name=tool_name,
                status="completed" if is_completed else "failed",
                arguments=rec.get("arguments", {}),
                result_summary=result_summary,
                latency_ms=latency_ms,
                error=rec.get("error_message"),
            ))

            if tool_name == "search_routing_tool" and result:
                raw = _json.loads(result) if isinstance(result, str) else result
                if isinstance(raw, list):
                    for i, item in enumerate(raw, 1):
                        documents.append(
                            ContextDocument(
                                id=f"D{i}",
                                title=item.get("title", ""),
                                content=item.get("content", ""),
                                url=item.get("url"),
                                score=0.0,
                                metadata={"source": "search_routing_tool"},
                            )
                        )
        except Exception:
            pass
```

Then change the return to include `tool_calls`:
```python
extra["tool_calls"] = tool_calls
citations = [doc.citation for doc in documents]
return answer, citations, documents, intent, extra
```

- [ ] **Step 8: Pass `tool_calls` into `AgentExperienceResponse` in `run_agent`**

In `_run_agent_impl` (or `run_agent`), find where `_run_auto_routed` is awaited and `AgentExperienceResponse` is constructed. After the `await`, extract `tool_calls`:

```python
answer, citations, documents, intent, extra = await _run_auto_routed(...)
tool_calls = extra.pop("tool_calls", [])
# ... existing session/message building ...
return AgentExperienceResponse(
    ...
    intent=intent,
    tool_calls=tool_calls,
)
```

- [ ] **Step 9: Run the backend tests**

```bash
cd /Users/linghuang/Git/Agentic-Search/.worktrees/feat-tool-trace
pytest tests/unit/servers/web/test_tool_trace.py -v 2>&1 | tail -20
```

Expected: all 6 tests pass.

- [ ] **Step 10: Run full Python unit suite to check no regressions**

```bash
pytest tests/unit/ -x -q 2>&1 | tail -5
```

Expected: 1809+ passed.

- [ ] **Step 11: Commit**

```bash
git add src/agents/state.py src/agents/tool_calling.py \
        src/internal/servers/web/app.py \
        tests/unit/servers/web/test_tool_trace.py
git commit -m "feat(backend): add ToolCallView trace parsing; expose tool_calls in AgentExperienceResponse"
```

---

## Task 2: Frontend types + `ToolCallTracePanel` component

**Files:**
- Modify: `web/src/types.ts`
- Create: `web/src/components/ToolCallTracePanel.tsx`
- Create: `web/src/components/__tests__/ToolCallTracePanel.test.tsx`

- [ ] **Step 1: Write failing component tests**

Create `web/src/components/__tests__/ToolCallTracePanel.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ToolCallTracePanel } from "../ToolCallTracePanel";
import type { ToolCallTraceView } from "../../types";

const completedCall: ToolCallTraceView = {
  tool_name: "search_routing_tool",
  status: "completed",
  arguments: { query: "FAISS" },
  result_summary: "3 items",
  latency_ms: 123,
  error: null,
};

const failedCall: ToolCallTraceView = {
  tool_name: "bad_tool",
  status: "failed",
  arguments: {},
  result_summary: "",
  latency_ms: 45,
  error: "Connection refused",
};

describe("ToolCallTracePanel", () => {
  it("renders nothing when calls is empty", () => {
    const { container } = render(<ToolCallTracePanel calls={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders a card for each call", () => {
    render(<ToolCallTracePanel calls={[completedCall, failedCall]} />);
    expect(screen.getByText("search_routing_tool")).toBeInTheDocument();
    expect(screen.getByText("bad_tool")).toBeInTheDocument();
  });

  it("completed call shows green checkmark", () => {
    render(<ToolCallTracePanel calls={[completedCall]} />);
    const status = document.querySelector(".tool-trace-status--ok");
    expect(status).not.toBeNull();
    expect(status?.textContent).toBe("✓");
  });

  it("failed call has tool-trace-card--failed class and shows error", () => {
    render(<ToolCallTracePanel calls={[failedCall]} />);
    expect(document.querySelector(".tool-trace-card--failed")).not.toBeNull();
    expect(screen.getByText("Connection refused")).toBeInTheDocument();
  });

  it("shows latency in ms", () => {
    render(<ToolCallTracePanel calls={[completedCall]} />);
    expect(screen.getByText("123 ms")).toBeInTheDocument();
  });

  it("renders arguments as JSON", () => {
    render(<ToolCallTracePanel calls={[completedCall]} />);
    expect(screen.getByText(/FAISS/)).toBeInTheDocument();
  });

  it("shows result_summary for completed call", () => {
    render(<ToolCallTracePanel calls={[completedCall]} />);
    expect(screen.getByText("3 items")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run failing tests**

```bash
cd /Users/linghuang/Git/Agentic-Search/.worktrees/feat-tool-trace/web
npm test -- --run src/components/__tests__/ToolCallTracePanel.test.tsx 2>&1 | tail -15
```

Expected: tests FAIL (module not found / no export).

- [ ] **Step 3: Add `ToolCallTraceView` to `web/src/types.ts`**

Append to the end of `web/src/types.ts`:

```typescript
export interface ToolCallTraceView {
  tool_name: string;
  status: "completed" | "failed";
  arguments: Record<string, unknown>;
  result_summary: string;
  latency_ms: number;
  error: string | null;
}
```

Also add `tool_calls?` to `AgentExperienceResponse`:
```typescript
export interface AgentExperienceResponse {
  session_id: string;
  answer: string;
  citations: string[];
  documents: SourceDocumentView[];
  messages: ChatMessageView[];
  intent?: "search" | "chat" | "tool";
  tool_calls?: ToolCallTraceView[];   // add this
}
```

And add `tool_calls?` to `SSEDoneEvent`:
```typescript
export interface SSEDoneEvent {
  type: "done";
  session_id: string;
  citations: string[];
  documents: SourceDocumentView[];
  intent?: "search" | "chat" | "tool";
  tool_calls?: ToolCallTraceView[];   // add this
}
```

- [ ] **Step 4: Create `web/src/components/ToolCallTracePanel.tsx`**

```tsx
import { memo } from "react";
import { Wrench } from "lucide-react";
import type { ToolCallTraceView } from "../types";

interface ToolCallTracePanelProps {
  calls: ToolCallTraceView[];
}

export const ToolCallTracePanel = memo(function ToolCallTracePanel({
  calls,
}: ToolCallTracePanelProps) {
  if (calls.length === 0) return null;

  return (
    <section className="panel tool-trace-panel" aria-label="Tool calls">
      <div className="section-heading">
        <Wrench size={18} />
        <h2>Tool Calls</h2>
        <span className="count">{calls.length}</span>
      </div>
      <div className="tool-trace-list">
        {calls.map((call, i) => (
          <div
            key={i}
            className={`tool-trace-card${call.status === "failed" ? " tool-trace-card--failed" : ""}`}
          >
            <div className="tool-trace-header">
              <span className={`tool-trace-status ${call.status === "failed" ? "tool-trace-status--failed" : "tool-trace-status--ok"}`}>
                {call.status === "completed" ? "✓" : "✗"}
              </span>
              <strong className="tool-trace-name">{call.tool_name}</strong>
              <span className="tool-trace-latency">{call.latency_ms} ms</span>
            </div>

            <div className="tool-trace-section">
              <span className="tool-trace-label">Arguments</span>
              <code className="tool-trace-code">
                {JSON.stringify(call.arguments, null, 2)}
              </code>
            </div>

            {call.status === "completed" ? (
              <div className="tool-trace-section">
                <span className="tool-trace-label">Result</span>
                <code className="tool-trace-code">{call.result_summary || "—"}</code>
              </div>
            ) : (
              <div className="tool-trace-section">
                <span className="tool-trace-label">Error</span>
                <code className="tool-trace-code tool-trace-code--error">
                  {call.error || "Unknown error"}
                </code>
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
});
```

- [ ] **Step 5: Run component tests**

```bash
cd /Users/linghuang/Git/Agentic-Search/.worktrees/feat-tool-trace/web
npm test -- --run src/components/__tests__/ToolCallTracePanel.test.tsx 2>&1 | tail -15
```

Expected: all 7 tests pass.

- [ ] **Step 6: Commit**

```bash
git add web/src/types.ts \
        web/src/components/ToolCallTracePanel.tsx \
        web/src/components/__tests__/ToolCallTracePanel.test.tsx
git commit -m "feat(frontend): add ToolCallTraceView type and ToolCallTracePanel component"
```

---

## Task 3: Wire `App.tsx` + SSE streaming + CSS

**Files:**
- Modify: `web/src/App.tsx`
- Modify: `web/src/styles.css`

- [ ] **Step 1: Read `web/src/App.tsx`** to understand current structure before editing.

- [ ] **Step 2: Update `App.tsx`**

Add import at the top:
```typescript
import { ToolCallTracePanel } from "./components/ToolCallTracePanel";
import type { ToolCallTraceView } from "./types";
```

Add state in the component:
```typescript
const [toolCalls, setToolCalls] = useState<ToolCallTraceView[]>([]);
```

In `handleNewSession` (the function that resets state), add:
```typescript
setToolCalls([]);
```

In the `streamAgent` loop, find the `} else if (event.type === "done") {` block and add:
```typescript
if (event.tool_calls) setToolCalls(event.tool_calls);
```

In the JSX, inside `.results-layout`, after `<AnswerPanel ...>` and before the Sources section, add:
```tsx
{intent === "tool" && toolCalls.length > 0 && (
  <ToolCallTracePanel calls={toolCalls} />
)}
```

- [ ] **Step 3: Append CSS to `web/src/styles.css`**

Append at the end:

```css
/* ── Tool call trace panel ───────────────────────────────────────────────────── */

.tool-trace-panel {
  margin-top: 12px;
}

.tool-trace-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.tool-trace-card {
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 10px 12px;
  background: #0f172a;
}

.tool-trace-card--failed {
  border-color: #7f1d1d;
  background: #0f0a0a;
}

.tool-trace-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.tool-trace-status--ok   { color: #22c55e; font-size: 0.8rem; }
.tool-trace-status--failed { color: #ef4444; font-size: 0.8rem; }

.tool-trace-name {
  color: #e2e8f0;
  font-size: 0.8rem;
}

.tool-trace-latency {
  margin-left: auto;
  color: #94a3b8;
  font-size: 0.65rem;
  font-variant-numeric: tabular-nums;
}

.tool-trace-section {
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 4px;
  padding: 6px 8px;
  margin-top: 6px;
}

.tool-trace-label {
  display: block;
  color: #64748b;
  font-size: 0.62rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 3px;
}

.tool-trace-code {
  color: #94a3b8;
  font-size: 0.68rem;
  font-family: monospace;
  white-space: pre-wrap;
  word-break: break-all;
}

.tool-trace-code--error { color: #ef4444; }
```

- [ ] **Step 4: Run full frontend test suite**

```bash
cd /Users/linghuang/Git/Agentic-Search/.worktrees/feat-tool-trace/web
npm test -- --run 2>&1 | tail -10
```

Expected: all tests pass (65+).

- [ ] **Step 5: Run typecheck**

```bash
cd /Users/linghuang/Git/Agentic-Search/.worktrees/feat-tool-trace/web
npm run typecheck 2>&1 | grep "error TS" | head -5 || echo "clean"
```

Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add web/src/App.tsx web/src/styles.css
git commit -m "feat(frontend): wire toolCalls state in App.tsx; add tool-trace CSS"
```

---

## Task 4: Also forward `tool_calls` in the SSE streaming path

**Files:**
- Modify: `src/internal/servers/web/app.py`

The `stream_agent` SSE endpoint emits a `done` event from `_run_agent_impl` result. Currently the `done` event doesn't include `tool_calls`. Fix:

- [ ] **Step 1: Find the SSE `done` yield in `stream_agent`**

In `app.py`, find `stream_agent` and the line:
```python
yield _sse({"type": "done", "session_id": result.session_id,
            "citations": result.citations,
            "documents": [d.model_dump() for d in result.documents],
            "intent": result.intent})
```

Add `"tool_calls": [tc.model_dump() for tc in result.tool_calls]`:
```python
yield _sse({"type": "done", "session_id": result.session_id,
            "citations": result.citations,
            "documents": [d.model_dump() for d in result.documents],
            "intent": result.intent,
            "tool_calls": [tc.model_dump() for tc in result.tool_calls]})
```

- [ ] **Step 2: Run the full Python unit suite**

```bash
pytest tests/unit/ -x -q 2>&1 | tail -5
```

Expected: 1815+ passed.

- [ ] **Step 3: Commit**

```bash
git add src/internal/servers/web/app.py
git commit -m "feat(backend): include tool_calls in SSE done event"
```

---

## Task 5: Final verification

- [ ] **Step 1: Full frontend test suite**

```bash
cd /Users/linghuang/Git/Agentic-Search/.worktrees/feat-tool-trace/web
npm test -- --run 2>&1 | tail -8
```

- [ ] **Step 2: Typecheck**

```bash
npm run typecheck 2>&1 | grep "error TS" | head -5 || echo "clean"
```

- [ ] **Step 3: Full Python unit suite**

```bash
cd /Users/linghuang/Git/Agentic-Search/.worktrees/feat-tool-trace
pytest tests/unit/ -x -q 2>&1 | tail -5
```

Expected: 1815+ passed.
