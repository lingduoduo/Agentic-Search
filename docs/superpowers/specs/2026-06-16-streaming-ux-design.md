# Streaming UX Design Spec

**Date:** 2026-06-16
**Status:** Draft

---

## 1. Goals & Success Criteria

### Problem

`/api/agent/stream` exists but is a stub — it calls the blocking `run_agent()` internally and emits only `answer` + `done` SSE events after the full response is ready. Users see a blank screen for the entire agent execution time (often 5–30 seconds for multi-turn search).

`streamAgent()` already exists in `web/src/api.ts` and correctly parses SSE events, but `App.tsx` still calls `runAgent()` (the blocking version).

### Success Criteria

- `/api/agent/stream` emits one `progress` SSE event after each agent turn completes
- `App.tsx` uses `streamAgent()` exclusively; `runAgent()` is removed from the main submit path
- During agent execution, `AnswerPanel` shows a live progress log with turn number and tool name
- After the agent finishes, the log collapses to a one-line summary ("3 turns · 8 docs retrieved") with a "show reasoning ▸" toggle
- The non-streaming `/api/agent` endpoint is unchanged (zero regression for callers that need a single JSON blob)
- `PlainGenerationLoop` (chat mode, single turn) skips `on_turn` — no visual regression for chat

---

## 2. Architecture

```
Browser                           FastAPI /api/agent/stream
  │                                          │
  │  POST /api/agent/stream                  │
  │─────────────────────────────────────────▶│
  │                                          │  asyncio.Queue()
  │                                          │  asyncio.create_task(run_agent(..., on_turn=queue.put))
  │◀── data: {"type":"progress","turn":1,"text":"search_routing_tool · 5 docs"} ──│
  │◀── data: {"type":"progress","turn":2,"text":"search_routing_tool · 3 docs"} ──│
  │◀── data: {"type":"progress","turn":3,"text":"writing answer…"} ────────────────│
  │◀── data: {"type":"answer","text":"Dense retrieval is…"} ──────────────────────│
  │◀── data: {"type":"done","session_id":"…","citations":[…],"documents":[…]} ────│
```

The agent loop receives an `on_turn` async callback. It calls it after each completed turn. The stream endpoint converts each call into an SSE `progress` event. This is the only coupling point — the rest of the agent loop is unchanged.

---

## 3. Backend Changes

### 3.1 `AgentLoopBase.run()` — add `on_turn` parameter

**File:** `src/agents/base.py`

Add to the `run()` abstract signature:

```python
from collections.abc import Awaitable, Callable

OnTurnCallback = Callable[[int, str | None, int], Awaitable[None]]
# args: (turn_number: int, tool_name: str | None, doc_count: int)

class AgentLoopBase:
    async def run(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        *,
        on_turn: OnTurnCallback | None = None,
    ) -> AgentLoopOutput:
        raise NotImplementedError
```

### 3.2 `SearchAgentLoop.run()` — call `on_turn` each turn

**File:** `src/agents/search.py`

After each turn's search results are collected (at the bottom of the `for turn in range(cfg.max_turns):` loop, before the `answer_tag` break check), add:

```python
if on_turn is not None:
    doc_count = len(search_results)  # results collected this turn
    tool_name = "search_routing_tool" if search_results else None
    await on_turn(turn + 1, tool_name, doc_count)
```

For the final turn (when `<answer>` tag fires), call before break:

```python
if on_turn is not None:
    await on_turn(num_turns, None, 0)  # tool_name=None signals "writing answer"
```

### 3.3 `ToolAgentLoop.run()` — call `on_turn` after each tool execution

**File:** `src/agents/tool_calling.py`

After each `tool_result` is collected in the `while True:` loop:

```python
if on_turn is not None and tool_results:
    last = tool_results[-1]
    await on_turn(assistant_turns, last.tool_name, 0)
```

### 3.4 `PlainGenerationLoop.run()` — no change

Single-turn, no `on_turn` calls needed. The signature change (kwarg) is backward-compatible — existing callers that don't pass `on_turn` are unaffected.

### 3.5 `/api/agent/stream` — true async SSE generator

**File:** `src/internal/servers/web/app.py`

Replace the current stub with an async generator that uses a `Queue` as the bridge:

```python
@app.post("/api/agent/stream")
async def stream_agent(
    request: AgentExperienceRequest,
    http_request: Request,
) -> StreamingResponse:
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    async def on_turn(turn: int, tool_name: str | None, doc_count: int) -> None:
        if tool_name:
            text = f"{tool_name} · {doc_count} docs"
        else:
            text = "writing answer…"
        await queue.put({"type": "progress", "turn": turn, "text": text})

    async def _generate():
        task = asyncio.create_task(run_agent(request, http_request, on_turn=on_turn))
        try:
            while not task.done():
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.05)
                    yield _sse(item)
                except asyncio.TimeoutError:
                    continue
            # Drain remaining progress events
            while not queue.empty():
                yield _sse(queue.get_nowait())
            # Emit final answer + done
            result: AgentExperienceResponse = task.result()
            yield _sse({"type": "answer", "text": result.answer})
            yield _sse({
                "type": "done",
                "session_id": result.session_id,
                "citations": result.citations,
                "documents": [d.model_dump() for d in result.documents],
                "intent": result.intent,
            })
        except Exception as exc:
            yield _sse({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

`run_agent` receives `on_turn` and passes it through to whichever agent loop it dispatches to. The signature of `run_agent` gains `on_turn: OnTurnCallback | None = None`.

---

## 4. Frontend Changes

### 4.1 New type `ProgressStep`

**File:** `web/src/types.ts`

```typescript
export interface ProgressStep {
  turn: number;
  text: string;      // e.g. "search_routing_tool · 5 docs"
}
```

Add to `SSEDoneEvent`:

```typescript
export interface SSEDoneEvent {
  type: "done";
  session_id: string;
  citations: string[];
  documents: SourceDocumentView[];
  intent?: "search" | "chat" | "tool";  // add this
}
```

### 4.2 `App.tsx` — replace `runAgent` with `streamAgent`

**File:** `web/src/App.tsx`

Add two state variables:

```typescript
const [progressSteps, setProgressSteps] = useState<ProgressStep[]>([]);
const [completedSteps, setCompletedSteps] = useState<ProgressStep[]>([]);
```

In `handleSubmit`, clear both before each new request:

```typescript
setProgressSteps([]);
setCompletedSteps([]);
```

Replace the `runAgent` call with the `streamAgent` async generator:

```typescript
for await (const event of streamAgent(agentRequest, { signal: controller.signal })) {
  if (event.type === "progress") {
    setProgressSteps((prev) => [...prev, { turn: event.turn, text: event.text }]);
  } else if (event.type === "answer") {
    setStreamingAnswer(event.text);
  } else if (event.type === "done") {
    setSessionId(event.session_id);
    setCitations(event.citations);
    setDocuments(event.documents);
    if (event.intent) setIntent(event.intent as "search" | "chat" | "tool");
    setAnswer(streamingAnswer);          // streamingAnswer accumulated from answer events
    setCompletedSteps(progressSteps);   // snapshot for collapsed summary
    setProgressSteps([]);               // clear live log
  } else if (event.type === "error") {
    setError(event.detail);
  }
}
```

Pass both to `AnswerPanel`:

```tsx
<AnswerPanel
  answer={streamingAnswer || answer}
  citations={citations}
  intent={intent}
  documentCount={documents.length}
  progressSteps={progressSteps}
  completedSteps={completedSteps}
/>
```

Remove the `runAgent` import.

### 4.3 `AnswerPanel.tsx` — collapsible progress log

**File:** `web/src/components/AnswerPanel.tsx`

Add props:

```typescript
interface AnswerPanelProps {
  answer: string;
  citations: string[];
  intent?: "search" | "chat" | "tool";
  documentCount?: number;
  toolCallCount?: number;
  progressSteps?: ProgressStep[];   // live steps while agent is running (empty when done)
  completedSteps?: ProgressStep[];  // snapshot of steps after done, drives collapsed summary
}
```

Render the `ProgressLog` sub-component above the answer:

```tsx
function ProgressLog({
  steps,
  completedSteps,
}: {
  steps: ProgressStep[];
  completedSteps: ProgressStep[];
}) {
  const [expanded, setExpanded] = useState(false);

  // Live log while agent is running
  if (steps.length > 0) {
    return (
      <div className="progress-log">
        <div className="progress-log-header">Agent reasoning</div>
        {steps.map((s) => (
          <div key={s.turn} className={`progress-step ${s.text.includes("writing") ? "active" : "done"}`}>
            {s.text.includes("writing") ? "⟳" : "✓"} Turn {s.turn} · {s.text}
          </div>
        ))}
      </div>
    );
  }

  // Collapsed summary after done
  if (completedSteps.length > 0) {
    const n = completedSteps.length;
    if (!expanded) {
      return (
        <button className="progress-summary" onClick={() => setExpanded(true)}>
          <span>✓ {n} {n === 1 ? "turn" : "turns"}</span>
          <span className="show-reasoning">show reasoning ▸</span>
        </button>
      );
    }
    return (
      <div className="progress-log">
        <div className="progress-log-header">
          Agent reasoning
          <button onClick={() => setExpanded(false)} className="collapse-btn">▾ hide</button>
        </div>
        {completedSteps.map((s) => (
          <div key={s.turn} className="progress-step done">
            ✓ Turn {s.turn} · {s.text}
          </div>
        ))}
      </div>
    );
  }

  return null;
}
```

`steps` is the live `progressSteps` from App.tsx; `completedSteps` is the snapshot saved on `done`. The component never needs its own "isDone" flag — the two arrays drive the three states (running / collapsed / expanded) cleanly.

---

## 5. CSS additions

**File:** `web/src/styles.css`

```css
.progress-log {
  border: 1px solid #1e3a5f;
  border-radius: 6px;
  padding: 10px 12px;
  background: #0d1f33;
  margin-bottom: 12px;
  font-size: 0.75rem;
}

.progress-log-header {
  color: #64748b;
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 6px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.progress-step.done  { color: #22c55e; }
.progress-step.active { color: #f59e0b; }
.progress-step + .progress-step { margin-top: 3px; }

.progress-summary {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 10px;
  background: #0d1f33;
  border: 1px solid #1e3a5f;
  border-radius: 4px;
  margin-bottom: 10px;
  cursor: pointer;
  color: #22c55e;
  font-size: 0.72rem;
  width: 100%;
  text-align: left;
}

.show-reasoning {
  color: #475569;
  font-size: 0.65rem;
  margin-left: auto;
}

.collapse-btn {
  background: none;
  border: none;
  color: #475569;
  cursor: pointer;
  font-size: 0.7rem;
  padding: 0;
}
```

---

## 6. Error Handling

| Scenario | Behavior |
|---|---|
| `error` SSE event | Set `error` state; clear `progressSteps` |
| Network disconnect mid-stream | `AbortController` cancels the fetch; `finally` clears loading state |
| Agent throws before first `progress` event | SSE `error` event emitted; frontend shows error banner |
| `on_turn` callback itself throws | Caught in `_generate()` try/except; emits `error` SSE event |

---

## 7. Testing Strategy

- **`tests/unit/agents/test_on_turn_callback.py`** — monkeypatch `SearchAgentLoop._step`, assert `on_turn` is called once per turn with correct `(turn, tool_name, doc_count)` args; same for `ToolAgentLoop`
- **`tests/unit/servers/web/test_stream_agent.py`** — `TestClient` + `httpx` streaming: assert `progress` events arrive before `done`; assert existing `/api/agent` still returns a plain JSON blob
- **Frontend:** `web/src/components/__tests__/AnswerPanel.test.tsx` — render with `progressSteps=[{turn:1,text:"search…"}]`, assert log is visible; render with `progressSteps=[]` + `totalTurns=2`, assert collapsed summary shows

---

## 8. File Map

| Action | Path | Responsibility |
|---|---|---|
| **Modify** | `src/agents/base.py` | Add `OnTurnCallback` type alias; add `on_turn` kwarg to abstract `run()` |
| **Modify** | `src/agents/search.py` | Call `on_turn` after each search turn and before final answer |
| **Modify** | `src/agents/tool_calling.py` | Call `on_turn` after each tool execution |
| **Modify** | `src/internal/servers/web/app.py` | Rebuild `stream_agent` with `asyncio.Queue`; thread `on_turn` through `run_agent` |
| **Modify** | `web/src/types.ts` | Add `ProgressStep`; add `intent` to `SSEDoneEvent` |
| **Modify** | `web/src/App.tsx` | Replace `runAgent` with `streamAgent` loop; add `progressSteps` state |
| **Modify** | `web/src/components/AnswerPanel.tsx` | Add `ProgressLog` sub-component with live/collapsed states |
| **Modify** | `web/src/styles.css` | Add `.progress-log`, `.progress-summary`, `.progress-step` styles |
| **Create** | `tests/unit/agents/test_on_turn_callback.py` | Unit tests for `on_turn` callback in all loops |
| **Create** | `tests/unit/servers/web/test_stream_agent.py` | HTTP-level SSE streaming tests |
| **Modify** | `web/src/components/__tests__/AnswerPanel.test.tsx` | Tests for live log and collapsed summary |

**Not changed:** `src/agents/plain.py` (single-turn, no progress callback needed), `/api/agent` endpoint, `api.ts` `streamAgent()` (already correct).
