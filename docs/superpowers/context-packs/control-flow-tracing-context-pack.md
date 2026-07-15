# Generated Context Pack

# Control Flow Tracing

## Sources

- [Specification: 2026-06-27-control-flow-tracing-design.md](../specs/2026-06-27-control-flow-tracing-design.md)
- [Plan: 2026-06-27-control-flow-tracing.md](../plans/2026-06-27-control-flow-tracing.md)

## Specification Context

### Goal

Add structured backend logging and a frontend debugging timeline so developers can linearly follow the automated agent workflow while it runs and inspect the same trace after completion or session reload.

The trace covers the improved control-flow components:

- Planner
- Loop Controller
- Search Tool
- Reranker Tool
- Evidence Judge
- Answer Generator

It reports operational decisions and outcomes, not private model reasoning.

### Agent and recorder unit tests

- sequence starts at 1 and increases exactly once per event;
- timestamps are UTC and events retain insertion order;
- detail allowlists and string bounds remove sensitive values;
- sink failure does not raise or stop in-memory collection;
- component start/completion/failure events carry correct duration and status;
- one live loop trajectory emits the expected component and decision order.

### Backend API tests

- non-streaming response includes the complete trace;
- SSE emits live `trace` events and repeats the final complete trace in `done`;
- assistant-message metadata persists the same sequence;
- session retrieval returns the trace metadata;
- progress events and responses without a trace remain compatible;
- full queries, documents, prompts, and secrets do not appear in serialized events or captured logs.

### Frontend tests

- live events render in sequence order;
- duplicate and out-of-order events deduplicate and sort;
- `done` replaces an incomplete live trace;
- completed traces collapse and expand;
- failed and active statuses have accessible text;
- session metadata restores the latest trace;
- malformed or absent legacy metadata renders no panel and no error.

### Verification

```bash
pytest tests/unit/test_control_flow_trace.py -v
pytest tests/unit/test_agent_loop.py tests/unit/test_chat_backend.py -v
cd web && npm test -- --run
cd web && npm run typecheck
ruff check src/agents src/internal/servers/web tests/unit
ruff format --check src/agents src/internal/servers/web tests/unit
```

Run the repository's full default `pytest` suite before completion.

### Out of Scope

- Human-in-the-loop approval or pause/resume behavior
- Distributed tracing systems such as OpenTelemetry
- Cross-service trace propagation into retrieval servers
- A graphical DAG or node-edge workflow visualization
- User-configurable trace verbosity
- Raw chain-of-thought display
- Removing the existing `progress` SSE event

## Implementation Plan Context

### Global Constraints

- Execute `docs/superpowers/plans/2026-06-27-agent-control-flow-consolidation.md` first; this plan instruments the authoritative component boundaries created there.
- Do not expose chain-of-thought, prompts, full queries, document text, credentials, headers, cookies, full tool payloads, or raw exception representations.
- Detail strings are capped at 256 characters and unknown detail keys are dropped.
- Event order is defined by a run-local sequence starting at 1, not by timestamps.
- Trace sink failures never alter agent execution or answer delivery.
- Preserve legacy responses, sessions, and the existing `progress` SSE event.
- Do not add human approval, pause/resume behavior, OpenTelemetry, or graphical workflow rendering.

---

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

_[Section compacted.]_

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

_[Section compacted.]_

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

_[Section compacted.]_

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

_[Section compacted.]_

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

_[Section compacted.]_

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

### Final Acceptance Checklist

- [ ] Backend logs and in-memory events share one schema and sequence.
- [ ] Search-agent components emit exactly one event per authoritative transition.
- [ ] SSE shows live events and `done` contains the complete trace.
- [ ] Non-streaming responses and assistant metadata contain the complete trace.
- [ ] Session reload restores the latest assistant trace.
- [ ] Frontend timeline is ordered, accessible, live, and collapsible after completion.
- [ ] Sensitive/raw content never enters trace details.
- [ ] Recorder or streaming sink failures never fail the agent run.
- [ ] Legacy clients, sessions, and progress events remain compatible.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
