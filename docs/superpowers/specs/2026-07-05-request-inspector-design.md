# Request Inspector — full-stage capture for the Dev Console — design

## Problem

When a query runs through `/api/agent`, its answer is the product of several
distinct stages — intent routing, retrieval, tool calls, LLM generation, and
final synthesis — but a developer cannot see what actually happened at each
stage. The existing control-flow trace (`ControlFlowRecorder`) is deliberately
**sanitized** to an allowlist of counts/decisions (`document_count`,
`evidence_score`, `sufficient`, `exit_reason`…). It does not carry the raw
material needed to debug behavior: the classifier's prompt and label, the exact
search queries and retrieved document bodies with scores, tool args/results, or
the LLM's raw prompt and completion.

## Goal

A **developer instrument** that captures the **full raw payload** of every stage
of a single request and presents it as **one request inspector** (all stages,
top-to-bottom, for a chosen run) inside the existing Dev Console. Recent runs are
inspectable via a **rolling in-memory history**. The capture path is entirely
gated behind the existing debug flag and is a **separate channel** from the
sanitized control-flow trace — the sanitized trace is untouched.

## Non-goals

- No change to the user-facing sanitized `ControlFlowRecorder` or its streaming.
- No durable persistence (snapshots live in memory, cleared on restart).
- No bespoke views per explicit `mode=`; the first cut targets the default
  auto-routed path (intent → RAG/search → final). Tool capture lands via the
  shared base dispatch but gets no dedicated mode-specific UI.
- No new determinism/routing behavior (covered separately by PR #374).

## Approach — ambient capture via `contextvars`

The five stages execute in different places (the router in `app.py`, retrieval
and generation inside the agent loops, the raw LLM call in the provider).
Threading a capture object through all of them would ripple across loop
signatures shared with the CLI. Instead, capture is **ambient**:

- A `ContextVar[RequestCapture | None]` (default `None`).
- Per request, when `debug_panels` is on, the handler calls `start_capture`,
  which creates a `RequestCapture` and sets the contextvar.
- Instrumentation points call a module-level `record_stage(...)` that **no-ops
  instantly when the contextvar is `None`** — one cheap `.get()`, zero cost on
  the hot path when the flag is off.

This spans the router + loops + provider + retrieval client with one-line emits
and no signature changes, and keeps raw capture cleanly separate from the
sanitized trace.

### New module: `src/internal/servers/web/request_capture.py`

```
_current: ContextVar[RequestCapture | None]  # default None

@dataclass
class StageRecord:
    stage: str            # "intent" | "search" | "tool" | "llm" | "final"
    label: str            # e.g. "classify_route", "synthesis"
    timestamp: float
    duration_ms: float | None
    payload: dict         # stage-specific RAW fields

@dataclass
class RequestCapture:
    request_id: str
    query: str
    created_at: float
    route: str | None
    route_degraded: str | None
    total_ms: float | None
    stages: list[StageRecord]
    def add(self, stage, label, payload, duration_ms=None) -> None
    def snapshot(self) -> dict

def start_capture(request_id: str, query: str) -> Token   # sets contextvar
def record_stage(stage, label, payload, duration_ms=None) -> None  # no-op if inactive
def capture_stage(stage, label)   # context manager: times block, records payload
def active() -> RequestCapture | None
```

Timing uses `time.monotonic()`; ids use `uuid4` — ordinary server code.

## Instrumentation points (5 stages)

Each is a one-line `record_stage(...)` at an existing choke point.

| Stage | Location | Raw payload |
| --- | --- | --- |
| intent | `classify_route` (`intent_routing.py`) + resolution in `_run_auto_routed` (`app.py`) | classifier prompt, raw LLM label, shortcut fired (`explicit_source`/`_is_bare_lookup`), final strategy, `route_degraded` |
| search | retrieval call in each loop (`AgenticRAGLoop`, `SearchAgentLoop` search_tool) | each search query, `top_k`, round #, retrieved docs `{id, title, text, score, source}` |
| tool | tool dispatch in `agents/core/base.py` | tool name, args, raw result |
| llm | `OpenAICompatibleLLM.complete`/stream (`providers.py`) **and** the local `manager.generate` call | model, prompt messages, raw completion, latency, token counts if present |
| final | `_finalize_response` (`app.py`) | answer, resolved citations, documents, surfaced intent + route/degradation |

The provider-layer **llm** emit catches every generation (classifier,
query-enhance, synthesis, answer) automatically. The local-model
`manager.generate` call is captured too so both backends are covered.

## Storage & API

- **Ring buffer** on `app.state`: `deque(maxlen=N)` (N default 20, env-overridable)
  plus a dict index by `request_id`. At the end of `_run_agent_impl`, if a
  capture is active, push its snapshot. Cleared on restart.
- **Endpoints** (in existing `debug_router.py`, already gated by `debug_panels`):
  - `GET /api/debug/requests` → `[{request_id, query, created_at, route, stage_count}]`, newest first.
  - `GET /api/debug/request/{id}` → full snapshot; 404 if evicted.
- The stream `done` event carries `request_id` so the panel auto-selects the run
  that just finished.

## Frontend

New `web/src/components/debug/RequestInspector.tsx`, added as a Dev Console
panel (alongside `RequestTracePanel`, `RetrievalLab`, etc.):

- **Left:** run list from `/api/debug/requests` (query + route + time).
- **Right:** selected run laid out top-to-bottom, **intent → search → tool → LLM
  → final**; each stage a collapsible card rendering its raw payload — docs table
  with scores, prompt/completion `<pre>` blocks, tool args/results JSON, timings.
- Reuses existing styling; no component library.

## Testing

- **Unit:** `record_stage` no-ops when inactive; `start_capture` + emits produce
  the expected snapshot; ring buffer evicts past N; endpoints return snapshot /
  404.
- **Integration:** one auto-routed request with `debug_panels` on yields a
  snapshot with all reached stages populated (raw prompt + retrieved docs
  present); with the flag off, no capture and zero added work.

## Verification / success criteria

- With `AGENTIC_SEARCH_DEBUG_PANELS` on, running a query and opening the Dev
  Console "Request Inspector" shows that run's intent, search, LLM, and final
  stages with full raw payloads; recent runs are selectable.
- With the flag off, `active()` is always `None`, no snapshot is stored, and no
  new endpoints do work.
- The sanitized `ControlFlowRecorder` output and existing tests are unchanged.
