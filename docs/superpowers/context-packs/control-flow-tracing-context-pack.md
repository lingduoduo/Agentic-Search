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

…

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

Add tests for UTC timestamps, invalid status rejection, defensive detail copying, and snapshot copying.

- [ ] **Step 2: Run tests to verify failure**

…

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

Extend one SearchAgentLoop trajectory to assert that `output.control_flow_trace` is ordered and non-empty after the recorder is wired.

- [ ] **Step 2: Run the focused test to verify failure**

Run: `pytest tests/unit/test_agent_loop.py -k control_flow_trace -v`

Expected: FAIL because `AgentLoopOutput` has no trace field.

…

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

…

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

Add an SSE assertion that `trace` appears before `answer`, and `done.control_flow_trace` equals the full ordered list.

- [ ] **Step 2: Run backend tests to verify failure**

…

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
