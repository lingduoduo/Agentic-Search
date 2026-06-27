# Structured Control-Flow Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose a safe, ordered search-agent control-flow trace through backend logs, live SSE, final responses, persisted session metadata, and a frontend linear timeline.

**Architecture:** A run-local `ControlFlowRecorder` is the single event source. Authoritative components record sanitized events; the same event objects are retained in `AgentLoopOutput`, streamed through the web callback, persisted in assistant-message metadata, and rendered by a focused React timeline.

**Tech Stack:** Python 3.11+, dataclasses, asyncio, FastAPI/Pydantic, SQLite JSON metadata, React 19, TypeScript, Vitest, pytest, Ruff.

## Global Constraints

- Execute `docs/superpowers/plans/2026-06-27-agent-control-flow-consolidation.md` first; this plan instruments the authoritative component boundaries created there.
- Do not expose chain-of-thought, prompts, full queries, document text, credentials, headers, cookies, full tool payloads, or raw exception representations.
- Detail strings are capped at 256 characters and unknown detail keys are dropped.
- Event order is defined by a run-local sequence starting at 1, not by timestamps.
- Trace sink failures never alter agent execution or answer delivery.
- Preserve legacy responses, sessions, and the existing `progress` SSE event.
- Do not add human approval, pause/resume behavior, OpenTelemetry, or graphical workflow rendering.

---

## Execution Order

1. Complete and verify the existing control-flow consolidation plan.
2. Execute Tasks 1–6 below in order.
3. Run the combined backend/frontend verification suite.

## File Map

- `src/agents/control_flow_trace.py` — event contract, sanitization, recorder, and safe live sink.
- `src/agents/base.py` — add the final trace to `AgentLoopOutput`.
- `src/agents/search.py` — create the recorder and attach its final snapshot.
- `src/agents/components/*.py` — emit events at authoritative component boundaries.
- `src/internal/servers/web/app.py` — API models, SSE trace events, persistence, and response mapping.
- `web/src/types.ts` — trace and SSE TypeScript contracts.
- `web/src/App.tsx` — live/final/session trace state.
- `web/src/components/ControlFlowTracePanel.tsx` — accessible ordered timeline.
- `web/src/styles.css` — timeline layout and status styles.

### Task 1: Build the safe run-local recorder

**Files:**
- Create: `src/agents/control_flow_trace.py`
- Create: `tests/unit/test_control_flow_trace.py`

**Interfaces:**
- Produces: `ControlFlowEvent`
- Produces: `ControlFlowRecorder.record(turn, component, action, status, duration_ms=None, details=None) -> ControlFlowEvent`
- Produces: `ControlFlowRecorder.snapshot() -> list[ControlFlowEvent]`
- Produces: optional `EventSink = Callable[[ControlFlowEvent], Awaitable[None]]`

- [ ] **Step 1: Write failing sequence and sanitization tests**

```python
def test_recorder_sequences_and_sanitizes_details() -> None:
    recorder = ControlFlowRecorder(request_id="req-1")
    first = recorder.record(
        turn=1,
        component="search_tool",
        action="vector_db_search",
        status="completed",
        details={
            "document_count": 5,
            "query": "secret query",
            "safe_message": "x" * 300,
        },
    )
    second = recorder.record(
        turn=1,
        component="evidence_judge",
        action="evidence_evaluated",
        status="completed",
        details={"evidence_score": 0.72, "sufficient": True},
    )

    assert [first.sequence, second.sequence] == [1, 2]
    assert "query" not in first.details
    assert len(first.details["safe_message"]) == 256
    assert recorder.snapshot() == [first, second]
```

Add tests for UTC timestamps, invalid status rejection, defensive detail copying, and snapshot copying.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/unit/test_control_flow_trace.py -v`

Expected: FAIL because `src.agents.control_flow_trace` does not exist.

- [ ] **Step 3: Implement the event and recorder**

Implement these exact public shapes:

```python
JsonValue = str | int | float | bool | None
EventSink = Callable[["ControlFlowEvent"], Awaitable[None]]

@dataclass(frozen=True, slots=True)
class ControlFlowEvent:
    sequence: int
    timestamp: str
    turn: int
    component: str
    action: str
    status: str
    duration_ms: int | None = None
    details: dict[str, JsonValue] = field(default_factory=dict)

class ControlFlowRecorder:
    def __init__(
        self,
        request_id: str,
        *,
        session_id: str | None = None,
        sink: EventSink | None = None,
    ) -> None:
        self._request_id = request_id
        self._session_id = session_id
        self._sink = sink
        self._events: list[ControlFlowEvent] = []
        self._sink_disabled = False

    def record(
        self,
        *,
        turn: int,
        component: str,
        action: str,
        status: str,
        duration_ms: int | None = None,
        details: Mapping[str, object] | None = None,
    ) -> ControlFlowEvent:
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"unsupported control-flow status: {status}")
        event = ControlFlowEvent(
            sequence=len(self._events) + 1,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            turn=turn,
            component=component,
            action=action,
            status=status,
            duration_ms=duration_ms,
            details=_sanitize_details(details or {}),
        )
        self._events.append(event)
        self._log(event)
        self._schedule_sink(event)
        return event

    def snapshot(self) -> list[ControlFlowEvent]:
        return list(self._events)
```

Use an allowlist containing the spec keys: `retriever`, `query_count`, `document_count`, `citation_count`, `evidence_score`, `sufficient`, `search_round`, `effective_budget`, `cache_hit_count`, `fallback`, `duplicate_count`, `overflow_count`, `error_category`, `safe_message`, `exit_reason`, and `decision`. Log `agent_control_flow` with structured `extra` fields.

For the async sink, schedule one task per event on the running loop, catch sink failures inside the task, disable only that sink, and retain every event in memory. When no event loop is running, retain/log without invoking the sink.

- [ ] **Step 4: Add and pass sink-failure tests**

```python
@pytest.mark.asyncio
async def test_sink_failure_does_not_break_recording() -> None:
    async def broken_sink(event: ControlFlowEvent) -> None:
        raise RuntimeError("sink unavailable")

    recorder = ControlFlowRecorder("req", sink=broken_sink)
    recorder.record(turn=1, component="planner", action="turn_parsed", status="completed")
    await asyncio.sleep(0)
    event = recorder.record(turn=1, component="planner", action="search_planned", status="decided")
    assert event.sequence == 2
    assert len(recorder.snapshot()) == 2
```

Run: `pytest tests/unit/test_control_flow_trace.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the recorder**

```bash
git add src/agents/control_flow_trace.py tests/unit/test_control_flow_trace.py
git commit -m "feat: add structured control-flow recorder"
```

### Task 2: Carry trace events in agent output

**Files:**
- Modify: `src/agents/base.py`
- Modify: `src/agents/search.py`
- Modify: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: `ControlFlowRecorder`
- Produces: `AgentLoopOutput.control_flow_trace: list[ControlFlowEvent]`
- Preserves: all existing constructors through a default empty list

- [ ] **Step 1: Write failing output compatibility tests**

```python
def test_agent_loop_output_defaults_to_empty_control_flow_trace() -> None:
    output = AgentLoopOutput(
        prompt_ids=[], response_ids=[], response_mask=[], num_turns=0,
        metrics={}, request_id="req",
    )
    assert output.control_flow_trace == []
```

Extend one SearchAgentLoop trajectory to assert that `output.control_flow_trace` is ordered and non-empty after the recorder is wired.

- [ ] **Step 2: Run the focused test to verify failure**

Run: `pytest tests/unit/test_agent_loop.py -k control_flow_trace -v`

Expected: FAIL because `AgentLoopOutput` has no trace field.

- [ ] **Step 3: Add the output field and run-local recorder**

In `AgentLoopOutput` add:

```python
control_flow_trace: list[ControlFlowEvent] = field(default_factory=list)
```

Add an optional trace sink to `SearchAgentLoop.run` without changing callers:

```python
async def run(
    self,
    messages: list[dict[str, Any]],
    sampling_params: dict[str, Any],
    *,
    on_turn: OnTurnCallback | None = None,
    on_trace: EventSink | None = None,
) -> AgentLoopOutput:
```

Create `ControlFlowRecorder(request_id, sink=on_trace)` once after request ID allocation and return `control_flow_trace=recorder.snapshot()`.

- [ ] **Step 4: Emit minimal loop lifecycle events**

Record only lifecycle events not owned by a component:

```python
recorder.record(
    turn=turn + 1,
    component="planner",
    action="format_recovery",
    status="decided",
    details={"decision": "retry"},
)
```

Do not record prompts, response text, or action contents.

- [ ] **Step 5: Run agent-loop regression tests**

Run: `pytest tests/unit/test_agent_loop.py -v`

Expected: PASS.

- [ ] **Step 6: Commit output propagation**

```bash
git add src/agents/base.py src/agents/search.py tests/unit/test_agent_loop.py
git commit -m "feat: expose agent control-flow trace"
```

### Task 3: Instrument authoritative components and decisions

**Files:**
- Modify: `src/agents/components/planner.py`
- Modify: `src/agents/components/search_tool.py`
- Modify: `src/agents/components/reranker_tool.py`
- Modify: `src/agents/components/evidence_judge.py`
- Modify: `src/agents/components/answer_generator.py`
- Modify: `src/agents/components/loop_controller.py`
- Modify: `src/agents/search.py`
- Modify: `tests/unit/test_components.py`
- Modify: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: `ControlFlowRecorder.record`
- Produces: the stable component/action vocabulary from the spec

- [ ] **Step 1: Add failing component event tests**

For each component, inject a recorder, execute one successful operation, and assert its event. Example:

```python
recorder = ControlFlowRecorder("req")
state = _state()
verdict = EvidenceJudge(evaluator=fake_evaluator, recorder=recorder).update_state(
    state, contexts, turn=2
)
event = recorder.snapshot()[-1]
assert (event.component, event.action, event.status) == (
    "evidence_judge", "evidence_evaluated", "completed"
)
assert event.details == {
    "evidence_score": verdict.score,
    "sufficient": verdict.is_sufficient,
    "document_count": sum(len(context.results) for context in contexts),
}
```

Cover planner decisions, vector/web/fallback search, rerank/skipped, controller continue/stop/answer outcomes, and citation resolution.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/unit/test_components.py tests/unit/test_loop_controller.py -v`

Expected: FAIL because components do not accept recorders or turn numbers.

- [ ] **Step 3: Add optional recorder dependencies**

Every component constructor accepts `recorder: ControlFlowRecorder | None = None`. Public methods that need ordering accept keyword-only `turn: int = 0`. Add one private helper per component:

```python
def _record(self, *, turn: int, action: str, status: str, **details: JsonValue) -> None:
    if self._recorder is not None:
        self._recorder.record(
            turn=turn,
            component="evidence_judge",
            action=action,
            status=status,
            details=details,
        )
```

Emit at the authoritative return/failure boundary. Measure durations with `time.perf_counter()` around retrieval/reranking/evaluation/generation only.

- [ ] **Step 4: Wire one recorder through the live loop**

Construct the run-scoped components with the same recorder. Pass the current one-based turn to component calls. For `LoopController`, either pass recorder/turn to decision methods or have `SearchAgentLoop` record immediately from returned typed decisions; choose one owner and assert there are no duplicate controller events.

- [ ] **Step 5: Add a full-order trajectory assertion**

```python
assert [(e.component, e.action) for e in output.control_flow_trace] == [
    ("planner", "search_planned"),
    ("search_tool", "vector_db_search"),
    ("evidence_judge", "evidence_evaluated"),
    ("loop_controller", "search_continued"),
    ("planner", "answer_planned"),
    ("loop_controller", "answer_accepted"),
    ("answer_generator", "citations_resolved"),
]
```

Run: `pytest tests/unit/test_components.py tests/unit/test_loop_controller.py tests/unit/test_agent_loop.py -v`

Expected: PASS.

- [ ] **Step 6: Commit component instrumentation**

```bash
git add src/agents/components src/agents/search.py tests/unit/test_components.py tests/unit/test_loop_controller.py tests/unit/test_agent_loop.py
git commit -m "feat: trace search-agent control decisions"
```

### Task 4: Add API, SSE, logging, and persistence propagation

**Files:**
- Modify: `src/internal/servers/web/app.py`
- Modify: `tests/unit/test_chat_backend.py`
- Modify: `tests/unit/servers/web/test_sse_streaming.py`

**Interfaces:**
- Produces: `ControlFlowEventView`
- Produces: `AgentExperienceResponse.control_flow_trace`
- Produces SSE: `{type: "trace", event: ControlFlowEventView}`

- [ ] **Step 1: Add failing response and persistence tests**

Assert that a mocked search-agent result with two events produces:

```python
assert response.control_flow_trace[0].sequence == 1
assistant = response.messages[-1]
assert assistant.metadata["control_flow_trace"] == [
    event.model_dump() for event in response.control_flow_trace
]
```

Add an SSE assertion that `trace` appears before `answer`, and `done.control_flow_trace` equals the full ordered list.

- [ ] **Step 2: Run backend tests to verify failure**

Run: `pytest tests/unit/test_chat_backend.py tests/unit/servers/web/test_sse_streaming.py -v`

Expected: FAIL because API and SSE models lack trace data.

- [ ] **Step 3: Add API models and mapping**

```python
class ControlFlowEventView(BaseModel):
    sequence: int
    timestamp: str
    turn: int
    component: str
    action: str
    status: str
    duration_ms: int | None = None
    details: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

class AgentExperienceResponse(BaseModel):
    # existing fields unchanged
    control_flow_trace: list[ControlFlowEventView] = Field(default_factory=list)
```

Create one `_trace_view(event)` mapper and use it for every explicit/auto search-agent response path. Non-search modes return the default empty list.

- [ ] **Step 4: Persist trace in assistant metadata**

Before `db.add_chat_message`, serialize once:

```python
trace_payload = [event.model_dump() for event in trace_views]
metadata = {
    # existing metadata
    "control_flow_trace": trace_payload,
}
```

Do not create a new table or migration.

- [ ] **Step 5: Stream live and final events**

Add a non-blocking callback for the agent run:

```python
async def on_trace(event: ControlFlowEvent) -> None:
    payload = {"type": "trace", "event": _trace_view(event).model_dump()}
    try:
        queue.put_nowait(payload)
    except asyncio.QueueFull:
        dropped_trace_events += 1
```

Pass `on_trace` only to loops that support it. Include the complete `control_flow_trace` in `done`. Log one warning with the dropped count after completion. Preserve existing `progress` events.

- [ ] **Step 6: Run backend regressions**

Run: `pytest tests/unit/test_chat_backend.py tests/unit/servers/web/test_sse_streaming.py tests/unit/test_db_store.py -v`

Expected: PASS.

- [ ] **Step 7: Commit backend propagation**

```bash
git add src/internal/servers/web/app.py tests/unit/test_chat_backend.py tests/unit/servers/web/test_sse_streaming.py
git commit -m "feat: stream and persist control-flow traces"
```

### Task 5: Render the live and persisted frontend timeline

**Files:**
- Modify: `web/src/types.ts`
- Modify: `web/src/App.tsx`
- Create: `web/src/components/ControlFlowTracePanel.tsx`
- Modify: `web/src/styles.css`
- Create: `web/src/components/__tests__/ControlFlowTracePanel.test.tsx`
- Modify: `web/src/components/__tests__/App.test.tsx`
- Modify: `web/src/__tests__/api.test.ts`

**Interfaces:**
- Produces: `ControlFlowEventView`, `SSETraceEvent`
- Produces: `<ControlFlowTracePanel events={events} live={isLoading} />`

- [ ] **Step 1: Add failing panel tests**

```tsx
it("renders events in sequence order and hides unknown details", () => {
  render(<ControlFlowTracePanel events={[event2, event1]} live={false} />);
  const items = screen.getAllByRole("listitem");
  expect(items[0]).toHaveTextContent("Planner");
  expect(items[1]).toHaveTextContent("Evidence judge");
  expect(screen.queryByText("secret")).not.toBeInTheDocument();
});

it("collapses completed traces and expands on request", async () => {
  render(<ControlFlowTracePanel events={[event1]} live={false} />);
  await userEvent.click(screen.getByRole("button", { name: /show control flow/i }));
  expect(screen.getByRole("list")).toBeVisible();
});
```

Also test active and failed accessible text, duration, and empty rendering.

- [ ] **Step 2: Run the panel tests to verify failure**

Run: `cd web && npm test -- --run src/components/__tests__/ControlFlowTracePanel.test.tsx`

Expected: FAIL because the component does not exist.

- [ ] **Step 3: Add TypeScript contracts**

```typescript
export interface ControlFlowEventView {
  sequence: number;
  timestamp: string;
  turn: number;
  component: string;
  action: string;
  status: "started" | "completed" | "decided" | "skipped" | "failed" | string;
  duration_ms: number | null;
  details: Record<string, string | number | boolean | null>;
}

export interface SSETraceEvent {
  type: "trace";
  event: ControlFlowEventView;
}
```

Add `control_flow_trace?: ControlFlowEventView[]` to response/done types and `SSETraceEvent` to the union.

- [ ] **Step 4: Implement the focused panel**

Use an ordered list and a fixed mapping for component/action labels and detail summaries. Never iterate arbitrary detail keys into JSX. When `events.length === 0`, return `null`. Default expanded while `live`; default collapsed after completion. Include `aria-live="polite"` on the live event region.

- [ ] **Step 5: Wire live, final, and session traces in App**

Use one helper:

```typescript
function upsertTrace(
  events: ControlFlowEventView[],
  event: ControlFlowEventView,
): ControlFlowEventView[] {
  return [...events.filter((item) => item.sequence !== event.sequence), event]
    .sort((a, b) => a.sequence - b.sequence);
}
```

Handle `trace` by upserting. Handle `done` by replacing with its authoritative trace. On session load, scan assistant messages from newest to oldest and accept `metadata.control_flow_trace` only when it is an array of shape-valid event objects. Reset trace on new query/session.

Render `<ControlFlowTracePanel>` below `AnswerPanel` and above `ToolCallTracePanel`.

- [ ] **Step 6: Run frontend tests and typecheck**

Run:

```bash
cd web && npm test -- --run
cd web && npm run typecheck
```

Expected: PASS.

- [ ] **Step 7: Commit frontend timeline**

```bash
git add web/src/types.ts web/src/App.tsx web/src/styles.css web/src/components/ControlFlowTracePanel.tsx web/src/components/__tests__/ControlFlowTracePanel.test.tsx web/src/components/__tests__/App.test.tsx web/src/__tests__/api.test.ts
git commit -m "feat: show linear agent control-flow timeline"
```

### Task 6: End-to-end compatibility and safety verification

**Files:**
- Modify: tests only when a discovered regression requires a focused test

**Interfaces:**
- Verifies: complete spec acceptance criteria

- [ ] **Step 1: Scan serialized traces for forbidden keys**

Run:

```bash
rg -n '"(query|prompt|document_text|token|cookie|headers|arguments|raw_output)"' src/agents/control_flow_trace.py tests/unit/test_control_flow_trace.py
```

Expected: no production allowlist entry for any forbidden key.

- [ ] **Step 2: Run focused backend verification**

```bash
pytest tests/unit/test_control_flow_trace.py tests/unit/test_components.py tests/unit/test_loop_controller.py tests/unit/test_agent_loop.py tests/unit/test_chat_backend.py tests/unit/servers/web/test_sse_streaming.py tests/unit/test_db_store.py -v
```

Expected: PASS.

- [ ] **Step 3: Run frontend verification**

```bash
cd web && npm test -- --run
cd web && npm run typecheck
```

Expected: PASS.

- [ ] **Step 4: Run static checks**

```bash
ruff check src/agents src/internal/servers/web tests/unit
ruff format --check src/agents src/internal/servers/web tests/unit
git diff --check
```

Expected: PASS.

- [ ] **Step 5: Run the full default backend suite**

Run: `pytest`

Expected: PASS; externally provisioned integration tests remain outside the default suite.

- [ ] **Step 6: Commit any verification-only adjustments**

If Step 1–5 required focused test or formatting changes:

```bash
git add src/agents src/internal/servers/web tests/unit web/src
git commit -m "test: verify control-flow trace integration"
```

If no files changed, do not create an empty commit.

## Final Acceptance Checklist

- [ ] Backend logs and in-memory events share one schema and sequence.
- [ ] Search-agent components emit exactly one event per authoritative transition.
- [ ] SSE shows live events and `done` contains the complete trace.
- [ ] Non-streaming responses and assistant metadata contain the complete trace.
- [ ] Session reload restores the latest assistant trace.
- [ ] Frontend timeline is ordered, accessible, live, and collapsible after completion.
- [ ] Sensitive/raw content never enters trace details.
- [ ] Recorder or streaming sink failures never fail the agent run.
- [ ] Legacy clients, sessions, and progress events remain compatible.
