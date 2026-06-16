# Tool Call Trace Panel Design Spec

**Date:** 2026-06-16
**Status:** Draft

---

## 1. Goals & Success Criteria

### Problem

When the agent runs in `tool` mode (`ToolAgentLoop`), the backend collects `action_trace` — a newline-delimited JSON log of every `ToolExecutionResult`. This trace is parsed internally to extract search documents but the full tool execution data (tool name, arguments, result, latency, errors) is discarded before the response reaches the frontend.

Users have no visibility into what tools ran, what they were called with, or whether they succeeded.

### Success Criteria

- `AgentExperienceResponse` includes `tool_calls: list[ToolCallView]` populated when `intent == "tool"`
- The frontend renders a `ToolCallTracePanel` below `AnswerPanel` only when `intent === "tool"`
- Each tool call card shows: tool name, status (✓ / ✗), arguments as JSON, result summary (first 200 chars or "N items" for lists), and latency in ms
- Failed calls render with a red border and the error message instead of result
- No change to the existing search (`intent="search"`) or chat (`intent="chat"`) paths

---

## 2. Architecture

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

## 3. Backend Changes

### 3.1 Add `arguments` to `ToolExecutionResult`

**File:** `src/agents/state.py`

`ToolExecutionResult` currently has no `arguments` field — they live on the caller's `FunctionCall` and are discarded after the call. Add:

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

**File:** `src/agents/tool_calling.py` — `_call_tool` method

Pass the parsed arguments when constructing `ToolExecutionResult`:

```python
return ToolExecutionResult(
    tool_name=tool_call.name,
    status=status,
    result=result,
    arguments=tool_call.parsed_arguments(),   # add this line
    performance=PerformanceMetrics(...),
    error_code=error_code,
    error_message=error_message,
)
```

`parsed_arguments()` returns `dict[str, Any]` and is already called earlier in the same method.

---

### 3.3 New `ToolCallView` Pydantic model

**File:** `src/internal/servers/web/app.py`

Add after `AgentExperienceResponse`:

```python
class ToolCallView(BaseModel):
    tool_name: str
    status: str          # "completed" | "failed"
    arguments: dict[str, object]
    result_summary: str  # first 200 chars of stringified result, or "N items" for lists
    latency_ms: int
    error: str | None = None
```

### 3.4 `AgentExperienceResponse` — add `tool_calls` field

```python
class AgentExperienceResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[str]
    documents: list[SourceDocumentView]
    messages: list[ChatMessageView]
    hook_metadata: dict[str, object] = Field(default_factory=dict)
    intent: str = "chat"
    tool_calls: list[ToolCallView] = Field(default_factory=list)   # add this
```

### 3.5 `_run_auto_routed` — parse `action_trace` into `ToolCallView` list

**File:** `src/internal/servers/web/app.py`

In the Tier 1 block, extend the existing `action_trace` parsing loop. Currently it only extracts `search_routing_tool` documents; extend it to also collect all tool calls:

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
            status = rec.get("status", "failed")
            # Normalise TaskStatus enum value (may be "TaskStatus.COMPLETED" or "completed")
            is_completed = "completed" in str(status).lower()
            result = rec.get("result")

            # Build result_summary
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

            # Existing document extraction (unchanged)
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

Return `tool_calls` from `_run_auto_routed` as part of the `extra` dict:

```python
extra["tool_calls"] = tool_calls
return answer, citations, documents, intent, extra
```

### 3.6 `run_agent` — pass `tool_calls` into response

In the `run_agent` function, where `_run_auto_routed` result is consumed, extract `tool_calls` from `extra` and pass it to `AgentExperienceResponse`:

```python
answer, citations, documents, intent, extra = await _run_auto_routed(...)
tool_calls = extra.pop("tool_calls", [])

return AgentExperienceResponse(
    session_id=session_id,
    answer=answer,
    citations=citations,
    documents=[...],
    messages=[...],
    intent=intent,
    tool_calls=tool_calls,
)
```

All other `AgentExperienceResponse` construction sites (non-tool paths) omit `tool_calls` — the default empty list applies.

---

## 4. Frontend Changes

### 4.1 New type `ToolCallTraceView`

**File:** `web/src/types.ts`

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

Add `tool_calls` to `AgentExperienceResponse`:

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

### 4.3 `App.tsx` — add `toolCalls` state and render panel

**File:** `web/src/App.tsx`

Add state:
```typescript
const [toolCalls, setToolCalls] = useState<ToolCallTraceView[]>([]);
```

In `handleSubmit`, after `runAgent` / `streamAgent` response:
```typescript
setToolCalls(response.tool_calls ?? []);
```

In `handleNewSession`, clear it:
```typescript
setToolCalls([]);
```

Render the panel in `results-layout`, between `AnswerPanel` and the Sources section:
```tsx
{intent === "tool" && toolCalls.length > 0 && (
  <ToolCallTracePanel calls={toolCalls} />
)}
```

Import at top: `import { ToolCallTracePanel } from "./components/ToolCallTracePanel";`
Import type: `import type { ToolCallTraceView } from "./types";`

---

## 5. CSS Additions

**File:** `web/src/styles.css`

```css
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

---

## 6. Error Handling

| Scenario | Behavior |
|---|---|
| `action_trace` is `None` | `tool_calls` stays `[]`; panel hidden |
| JSON line fails to parse | Caught by `except Exception: pass`; line skipped |
| `performance.execution_time` missing | `int(0.0 * 1000) = 0` ms shown |
| `result` is a JSON string (from search tool) | Parsed to list → "N items" summary |
| `arguments` missing from trace record | Defaults to `{}` — empty args block rendered |
| Non-tool intent path | `tool_calls` field defaults to `[]`; panel not rendered |

---

## 7. Testing Strategy

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

## 8. File Map

| Action | Path | Responsibility |
|---|---|---|
| **Modify** | `src/agents/state.py` | Add `arguments: dict` field to `ToolExecutionResult` |
| **Modify** | `src/agents/tool_calling.py` | Pass `tool_call.parsed_arguments()` into `ToolExecutionResult` |
| **Modify** | `src/internal/servers/web/app.py` | Add `ToolCallView` model; extend trace parsing; add `tool_calls` to response |
| **Modify** | `web/src/types.ts` | Add `ToolCallTraceView`; add `tool_calls?` to `AgentExperienceResponse` |
| **Create** | `web/src/components/ToolCallTracePanel.tsx` | New panel component |
| **Modify** | `web/src/App.tsx` | Add `toolCalls` state; render panel when `intent === "tool"` |
| **Modify** | `web/src/styles.css` | Add `.tool-trace-*` styles |
| **Create** | `tests/unit/servers/web/test_tool_trace.py` | Backend unit tests for trace parsing |
| **Create** | `web/src/components/__tests__/ToolCallTracePanel.test.tsx` | Frontend component tests |

**Not changed:** `src/agents/tool_calling.py` (trace format unchanged), all other agent loops, search/chat paths, `SourceGrid`, `SessionTimeline`, `AnswerPanel`.
