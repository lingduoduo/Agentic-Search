# Control-Flow Tracing Design

**Date:** 2026-06-27
**Status:** Approved for implementation planning
**Depends on:** `2026-06-27-agent-control-flow-consolidation-design.md`

## Goal

Add structured backend logging and a frontend debugging timeline so developers can linearly follow the automated agent workflow while it runs and inspect the same trace after completion or session reload.

The trace covers the improved control-flow components:

- Planner
- Loop Controller
- Search Tool
- Reranker Tool
- Evidence Judge
- Answer Generator

It reports operational decisions and outcomes, not private model reasoning.

## Current Problem

The streaming endpoint currently emits presentation strings such as `search_routing_tool · 5 docs` and `writing answer...`. They are useful as activity indicators but insufficient for debugging because they:

- do not identify every control-flow decision;
- have no stable schema, ordering field, status, duration, or structured details;
- exist only in the active frontend request;
- are not returned by the non-streaming endpoint;
- are not persisted as an inspectable trace on the assistant message;
- tempt clients to parse human-readable strings.

The agent loops also expose an `action_trace` string, but it contains model-produced action text rather than a sanitized operational record and is not an appropriate UI/debugging contract.

## Chosen Approach

Introduce one structured `ControlFlowEvent` contract and one run-local recorder. Every control-flow transition records an event once. The recorder:

1. assigns a strictly increasing sequence number;
2. sanitizes and bounds event details;
3. retains the ordered in-memory trace for the final result;
4. emits a structured server log entry;
5. invokes an optional async callback for live SSE delivery;
6. never raises into the agent workflow when logging or streaming fails.

The same serialized events flow through server logs, SSE, the final API response, assistant-message metadata, session history, and the frontend timeline.

## Alternatives Rejected

### Expand progress strings

This is a smaller patch, but it couples clients to copy, prevents reliable filtering, and leaves persistence and post-run diagnostics weak.

### Emit backend logs plus a final text blob

This supports server debugging but loses typed live frontend updates and forces consumers to parse text.

### Expose raw model action traces

Raw action output can contain prompts, reasoning, document text, or malformed content. It is noisy, unstable, and unsafe as a general debugging contract.

## Event Contract

Add a shared Python model in the agent layer and a matching Pydantic/API model plus TypeScript interface:

```python
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
```

Allowed status values are:

- `started`
- `completed`
- `decided`
- `skipped`
- `failed`

Component and action names are stable snake-case identifiers. Initial event vocabulary:

| Component | Actions |
|---|---|
| `planner` | `turn_parsed`, `search_planned`, `rerank_planned`, `answer_planned`, `format_recovery` |
| `loop_controller` | `search_continued`, `budget_exhausted`, `plateau_stopped`, `answer_accepted`, `answer_rejected`, `answer_forced` |
| `search_tool` | `vector_db_search`, `web_search`, `fallback_search`, `query_skipped` |
| `reranker_tool` | `rerank`, `rerank_skipped` |
| `evidence_judge` | `evidence_evaluated` |
| `answer_generator` | `answer_generated`, `citations_resolved` |

Example:

```json
{
  "sequence": 7,
  "timestamp": "2026-06-27T14:05:31.412Z",
  "turn": 2,
  "component": "evidence_judge",
  "action": "evidence_evaluated",
  "status": "completed",
  "duration_ms": 3,
  "details": {
    "evidence_score": 0.72,
    "sufficient": true,
    "document_count": 5
  }
}
```

`sequence`, not wall-clock timestamp, defines display order. Timestamps are UTC ISO-8601 strings for correlation with backend logs.

## Data-Safety Boundary

Trace details use an allowlist per component. Permitted values include:

- retriever name;
- query count, but not full query text;
- document and citation counts;
- evidence score and sufficiency boolean;
- search round and effective budget;
- cache-hit, fallback, duplicate, and overflow counts;
- stable error category and bounded safe message;
- exit reason and decision result.

The trace must not contain:

- chain-of-thought or `<think>` contents;
- system prompts or conversation transcripts;
- full search queries by default;
- document contents or fetched page text;
- credentials, headers, tokens, cookies, or connector configuration;
- complete tool arguments or raw tool output;
- exception representations that may embed payloads.

Strings in `details` are capped at 256 characters. Unknown detail keys are dropped. The recorder copies detail dictionaries so later mutation cannot change previously emitted events.

## Backend Architecture

### Recorder

Add `src/agents/control_flow_trace.py` containing:

- `ControlFlowEvent`
- `ControlFlowRecorder`
- component/action/status constants or enums
- detail sanitization
- a small timing context/helper for paired start/completion events

One recorder is created per agent run. It owns sequence assignment and the final immutable event snapshot. Components receive the recorder or a narrow `record(...)` callback through normal dependency injection; there is no global recorder and no context variable.

Recording is best-effort. If structured logging or the live callback fails, the recorder catches the error, writes one conventional warning, disables the failing sink for that run, and continues retaining in-memory events.

### Agent output

Add `control_flow_trace: list[ControlFlowEvent]` to `AgentLoopOutput`, defaulting to an empty list for every existing loop. `SearchAgentLoop` populates it from the recorder. Other loops may adopt the contract later without being required to emit events in this change.

The consolidated components emit events at their authoritative boundaries. `SearchAgentLoop.run()` may emit only lifecycle and format-recovery events that do not belong to a component. It must not duplicate component events.

### Web API

Add `ControlFlowEventView` and `control_flow_trace` to `AgentExperienceResponse`. The non-streaming `POST /api/agent` returns the complete list.

The streaming endpoint adds:

```json
{"type":"trace","event":{...ControlFlowEventView}}
```

The final `done` event also contains `control_flow_trace` as the authoritative complete copy. Clients deduplicate by `sequence`; this handles a lost live event or reconnection without inventing merge rules.

The existing `progress` SSE event remains temporarily for backward compatibility. The React frontend stops using it for search-agent control-flow display after adopting trace events. Removal of `progress` is a separate compatibility change.

### Persistence

Store the serialized trace in the existing assistant chat-message metadata under:

```json
{"control_flow_trace": [...]}
```

No database migration is needed because message metadata is already JSON. `ChatMessageView.metadata` returns it through existing session APIs.

Persist exactly the final recorder snapshot, not the sequence of SSE delivery attempts. Persistence failure follows current message-write error handling; trace-sink failure never changes the answer itself.

### Logging

Each event produces one structured log entry with the request/session correlation identifiers and event fields. Use the repository logger rather than printing. The human-readable message is constant (`agent_control_flow`); structured fields carry the useful values.

## Frontend Architecture

### Types and state

Add `ControlFlowEventView`, `SSETraceEvent`, and `control_flow_trace` fields to `web/src/types.ts`. `App.tsx` maintains one ordered trace array for the active result.

On a live `trace` event, upsert by `sequence` and sort ascending. On `done`, replace the array with the authoritative complete trace. On session load, select the latest assistant message's `metadata.control_flow_trace`, validate its shape defensively, and render it. Missing or malformed legacy metadata produces an empty trace without an error.

### Linear timeline

Add a focused `ControlFlowTracePanel` component below the answer and above tool-call details. It renders:

- sequence number;
- component label;
- action label;
- status;
- duration when present;
- one-line summary derived from known detail keys.

During execution the panel is expanded and the latest `started` event is marked active. After completion it collapses to a summary such as `8 control-flow events · 2 searches · evidence 0.72`; users can expand it to inspect the full ordered list.

Use semantic ordered-list markup. Status is communicated by text and icon, not color alone. The component does not render arbitrary detail values as HTML and does not display unknown keys.

The existing `ProgressLog` may continue serving non-search modes. The control-flow panel is shown whenever structured trace events exist, independent of intent routing labels.

## End-to-End Data Flow

1. The backend creates a recorder for the agent request.
2. Planner records the parsed/planned action.
3. Loop Controller records its decision.
4. Search, rerank, judge, and answer components record starts and outcomes at their authoritative boundaries.
5. The recorder assigns sequence, sanitizes details, logs the event, retains it, and sends it to the optional live callback.
6. The web streaming callback enqueues an SSE `trace` event.
7. React upserts the event and renders the linear timeline.
8. On completion, the backend persists the final trace in assistant-message metadata and returns it in both response styles.
9. React replaces live state with the final trace.
10. A later session load reconstructs the panel from assistant-message metadata.

## Failure Handling

- A trace callback or logging failure cannot fail the agent run.
- Queue saturation drops live trace delivery only; the final response and persisted trace remain complete. Emit one warning with the dropped-event count.
- Component failure records a sanitized `failed` event before existing fallback or termination behavior continues.
- Duplicate SSE events are harmless because the frontend upserts by sequence.
- Out-of-order network delivery is sorted by sequence.
- Unknown component/action/status values render with neutral labels for forward compatibility.
- Legacy API responses and sessions without `control_flow_trace` render normally.
- Aborted requests retain only events persisted by the existing successful message-write path; no partial trace record is added independently.

## Testing

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

## Success Criteria

1. Every authoritative search-agent component emits sanitized, structured control-flow events.
2. Events are strictly ordered and visible live through SSE.
3. Non-streaming and final streaming responses contain the complete trace.
4. The assistant message persists the same trace and session reload restores it.
5. The frontend presents a readable linear timeline during and after execution.
6. Trace failures never change agent execution or answer delivery.
7. No raw reasoning, prompts, document contents, secrets, or complete tool payloads enter the trace.
8. Existing clients and legacy sessions remain compatible.

## Out of Scope

- Human-in-the-loop approval or pause/resume behavior
- Distributed tracing systems such as OpenTelemetry
- Cross-service trace propagation into retrieval servers
- A graphical DAG or node-edge workflow visualization
- User-configurable trace verbosity
- Raw chain-of-thought display
- Removing the existing `progress` SSE event
