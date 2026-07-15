# Generated Context Pack

# Tool Call Trace

## Sources

- [Specification: 2026-06-16-tool-call-trace-design.md](../specs/2026-06-16-tool-call-trace-design.md)
- [Plan: 2026-06-17-tool-call-trace.md](../plans/2026-06-17-tool-call-trace.md)

## Specification Context

### 2. Architecture

```
ToolAgentLoop.run()
  └── action_trace = newline-delimited ToolExecutionResult.to_dict() JSON

_run_auto_routed() [Tier 1 path]
  └── parse action_trace lines → list[ToolCallView]
  └── return alongside (answer, citations, documents, intent, extra)

run_agent()
  └── AgentExperienceResponse(tool_calls=tool_calls, ...)

Frontend App.tsx
  └── toolCalls state ← response.tool_calls
  └── <ToolCallTracePanel calls={toolCalls} /> rendered when intent === "tool"
```

`ToolCallView` is a Pydantic model added to `app.py` alongside the existing response models. The parsing logic extends the existing `action_trace` loop in `_run_auto_routed`.

---

### 4.2 New `ToolCallTracePanel` component

**File:** `web/src/components/ToolCallTracePanel.tsx`

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
            className={`tool-trace-card ${call.status === "failed" ? "tool-trace-card--failed" : ""}`}
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

_[Section compacted.]_

### 7. Testing Strategy

- **`tests/unit/servers/web/test_tool_trace.py`** (new file)
  - Mock `_run_auto_routed` with an `action_trace` containing two completed calls and one failed call
  - Assert `AgentExperienceResponse.tool_calls` has length 3
  - Assert latency computed correctly from `performance.execution_time`
  - Assert list result → "N items", string result → truncated to 200 chars
  - Assert `error_message` mapped to `ToolCallView.error`

- **`web/src/components/__tests__/ToolCallTracePanel.test.tsx`** (new file)
  - Render with two completed calls — assert two cards, both green ✓
  - Render with one failed call — assert card has `tool-trace-card--failed` class, error text visible
  - Render with `calls=[]` — assert nothing rendered (`null`)

---

## Implementation Plan Context

### Task 1: Backend — `arguments` field + `ToolCallView` model + trace parsing

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

_[Section compacted.]_

### Task 2: Frontend types + `ToolCallTracePanel` component

**Files:**
- Modify: `web/src/types.ts`
- Create: `web/src/components/ToolCallTracePanel.tsx`
- Create: `web/src/components/__tests__/ToolCallTracePanel.test.tsx`

- [x] **Step 1: Write failing component tests**

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

_[Section compacted.]_

### Task 3: Wire `App.tsx` + SSE streaming + CSS

**Files:**
- Modify: `web/src/App.tsx`
- Modify: `web/src/styles.css`

- [x] **Step 1: Read `web/src/App.tsx`** to understand current structure before editing.

- [x] **Step 2: Update `App.tsx`**

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

- [x] **Step 3: Append CSS to `web/src/styles.css`**

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

_[Section compacted.]_

### Task 4: Also forward `tool_calls` in the SSE streaming path

**Files:**
- Modify: `src/internal/servers/web/app.py`

The `stream_agent` SSE endpoint emits a `done` event from `_run_agent_impl` result. Currently the `done` event doesn't include `tool_calls`. Fix:

- [x] **Step 1: Find the SSE `done` yield in `stream_agent`**

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

- [x] **Step 2: Run the full Python unit suite**

```bash
pytest tests/unit/ -x -q 2>&1 | tail -5
```

Expected: 1815+ passed.

- [x] **Step 3: Commit**

```bash
git add src/internal/servers/web/app.py
git commit -m "feat(backend): include tool_calls in SSE done event"
```

---

### Task 5: Final verification

- [x] **Step 1: Full frontend test suite**

```bash
cd /Users/linghuang/Git/Agentic-Search/.worktrees/feat-tool-trace/web
npm test -- --run 2>&1 | tail -8
```

- [x] **Step 2: Typecheck**

```bash
npm run typecheck 2>&1 | grep "error TS" | head -5 || echo "clean"
```

- [x] **Step 3: Full Python unit suite**

```bash
cd /Users/linghuang/Git/Agentic-Search/.worktrees/feat-tool-trace
pytest tests/unit/ -x -q 2>&1 | tail -5
```

Expected: 1815+ passed.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
