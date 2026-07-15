# Tool-agent human approval — Phase 1 design

**Status:** Approved design, ready for implementation planning

## Goal

Require the initiating user to approve each side-effecting `ToolAgentLoop`
invocation before it executes, while preserving the repository's current
in-memory agent loop and live SSE request model.

This phase adds a truthful in-process approval seam. It does not claim durable
pause or restart-safe resume; those require a persisted agent-run runtime and
belong in a separate design.

## Current architecture

The generic tool loop is request-local. `ToolAgentLoop.run()` keeps prompt IDs,
response masks, working messages, turn counters, and tool results in local
variables. Parsed tool calls are executed immediately and concurrently through
`_call_tool()`.

The web backend constructs a new loop for each automatic Tier-1 or explicit
`tool_agent` request. The non-streaming endpoint awaits the whole request. The
streaming endpoint runs the same coroutine in a task, forwards progress through
an in-memory queue, and persists the assistant message only after completion.

Several existing names suggest approval concepts but do not implement them:

- `Plan.requires_human_approval` has no runtime consumers.
- `EndpointPolicy.ASK` describes an absent external-app egress layer and has no
  call sites in this repository.
- notification constants for approval are declarations only.

The design therefore extends the working `Tool` and `ToolAgentLoop` paths. It
does not build on those inactive declarations.

## Scope

Phase 1 covers generic tools invoked by `ToolAgentLoop`, including the loop used
by automatic Tier-1 routing and explicit `tool_agent` mode.

Search-agent XML actions remain automated. Search and reranking are read-only
and use a different loop. Scheduled-task approval, external-app policy, direct
registry REST invocation, and MCP invocation are also outside this phase.

## Tool effect classification

Add a `ToolEffect` string enum in `src/tools/base.py`:

- `READ_ONLY`
- `SIDE_EFFECTING`
- `UNSPECIFIED`

Every `Tool` exposes an `effect` property. Function tools default to
`UNSPECIFIED`, and `FunctionTool` plus its decorator accept an explicit effect
override. The loop treats `UNSPECIFIED` exactly like `SIDE_EFFECTING`, so an
unclassified capability fails closed. Repository-owned search, RAG, calculator,
and other known-pure tools are migrated to explicit `READ_ONLY` declarations.

OpenAPI tools infer their effect from the operation method:

- `GET`, `HEAD`, and `OPTIONS` are read-only.
- `POST`, `PUT`, `PATCH`, and `DELETE` are side-effecting.

The effect is orchestration metadata. It is not added to the JSON function
schema shown to the model.

## Approval contracts

Add small, framework-neutral models near the tool-loop orchestration code:

```python
class ApprovalDecision(str, Enum):
    APPROVE = "approve"
    DENY = "deny"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class ToolApprovalRequest:
    approval_id: str
    tool_name: str
    arguments: dict[str, Any]
    created_at: datetime
    expires_at: datetime


ToolApprovalCallback = Callable[
    [ToolApprovalRequest], Awaitable[ApprovalDecision]
]
```

The internal request retains the exact parsed arguments needed for execution.
The web view produces a separate sanitized argument summary and never returns
secret-like keys or unrestricted nested payloads.

`ToolAgentLoop.run()` accepts an optional `on_approval` callback alongside its
existing `on_turn` callback. Approval is per invocation and is never remembered
for a run or chat session.

## Loop behavior

After parsing and truncating a generated tool-call batch to
`max_parallel_calls`, the loop preflights the whole batch before calling
`_call_tool()`:

1. Read-only calls are immediately eligible.
2. Each side-effecting or unspecified call receives a distinct approval
   request.
3. If no callback is installed, the call is denied without execution.
4. With a callback, all required decisions are awaited concurrently.
5. No tool in the generated batch executes until every gated call is resolved.
6. Approved and read-only calls execute through the existing `_call_tool()`
   path.
7. Denied and expired calls become `TaskStatus.SKIPPED` results with a stable,
   non-sensitive reason.

Skipped approval results are converted to tool-response messages and returned
to the model so it can explain the denial or choose another action. A skipped
result does not trigger the existing stop-on-failure branch. Genuine execution
failures retain the existing stop behavior.

The approval callback is advisory only for side-effecting calls. It cannot
approve a missing tool, bypass argument parsing, or turn a failed tool execution
into success.

## In-process approval broker

The web app owns one process-local `ToolApprovalBroker`. It maps opaque approval
IDs to pending entries containing:

- the initiating authenticated user ID,
- a future for the decision,
- creation and expiration times,
- the tool name and sanitized web view.

The broker provides three operations:

- `request(...)` registers an entry and awaits its future with a configurable
  timeout.
- `decide(...)` verifies ownership and resolves an unresolved future.
- `cancel(...)` removes and cancels an entry when its agent request ends.

The default timeout is 60 seconds and is configurable through the existing app
settings/environment-loading pattern. Timeout resolves as `EXPIRED`, removes
the entry, and never executes the tool.

Entries are removed in `finally` blocks after approval, denial, timeout, stream
disconnect, or agent failure. The broker stores no completed approval history.

## Web and SSE flow

The streaming endpoint supplies an `on_approval` callback to `_run_agent_impl`,
which threads it through both automatic Tier-1 routing and explicit
`tool_agent` mode.

When the loop requests approval, the callback:

1. registers the request with the broker,
2. enqueues an `approval_required` SSE event,
3. waits for the broker decision,
4. returns that decision to the still-running loop.

The event shape is:

```json
{
  "type": "approval_required",
  "approval": {
    "id": "opaque-id",
    "tool_name": "create_ticket",
    "arguments": {"project": "ENG", "title": "…"},
    "expires_at": "2026-06-27T22:30:00Z"
  }
}
```

Add `POST /api/agent/approvals/{approval_id}` with a body containing
`{"decision": "approve"}` or `{"decision": "deny"}`. The endpoint requires the
same authenticated user who initiated the agent request.

Responses are:

- `200` when a pending decision is accepted,
- `403` when the authenticated user does not own the request,
- `404` for an unknown or already-cleaned request,
- `409` when a concurrent decision has already resolved the future,
- `410` when the request has expired but has not yet been cleaned.

The original SSE connection stays open while approval is pending. After the
decision, normal progress, answer, and done events continue on that connection.

The non-streaming `/api/agent` endpoint and direct Python callers do not install
an interactive callback. Side-effecting calls fail closed as skipped. They do
not block waiting for a decision the caller cannot observe.

If a streaming request has no authenticated initiating user, side-effecting and
unspecified calls also fail closed as skipped; the server does not emit an
actionable approval event that nobody is authorized to decide.

## Frontend behavior

Extend the SSE union with an `approval_required` event and add an inline
approval card to the current progress area. The card shows:

- tool name,
- sanitized argument summary,
- expiration countdown,
- Approve and Deny buttons.

The frontend posts exactly one decision for each card. Both buttons disable
while the request is pending and remain disabled after the server accepts a
decision. A server rejection displays an inline error without pretending the
tool ran.

Approval cards are ephemeral request UI, not chat messages and not part of the
persisted control-flow timeline. The final tool trace records the resulting
completed, skipped, or failed tool execution status through existing output
mechanisms.

## Security and privacy

- Only the initiating authenticated user can decide.
- Approval IDs are random opaque values and are not authorization by
  themselves.
- The browser receives only sanitized argument summaries with bounded strings,
  bounded collection sizes, and secret-like keys removed.
- Exact arguments remain inside the running loop; the broker does not persist
  them.
- Approval cannot alter arguments. Any change requires a new model-generated
  invocation and a new approval.
- Missing callbacks, timeouts, malformed decisions, and broker failures all
  fail closed without executing side effects.

## Cancellation and concurrency

The current SSE generator cancels the agent task when the stream disconnects.
Cancellation must propagate through the approval callback, and its `finally`
block must remove the pending broker entry.

Each parallel side-effecting call gets its own approval ID. Decisions may arrive
in any order. The loop waits for all required decisions before executing any
call in the batch, preventing partial execution while the user is still
reviewing sibling actions.

The broker resolves a future at most once. Two concurrent decision requests
cannot cause duplicate tool execution because execution remains owned by the
single waiting loop coroutine.

## Observability

Log approval lifecycle events using approval ID, tool name, decision, duration,
and request/session identifiers where available. Never log raw arguments.

Add metrics for requested, approved, denied, expired, cancelled, and broker
errors. These metrics are additive and do not change existing tool-call metrics.

## Testing

### Tool and loop tests

- function-tool unspecified defaults and explicit effect classification,
- OpenAPI HTTP-method inference,
- read-only calls execute without a callback,
- side-effecting calls fail closed without a callback,
- approval executes once,
- denial and timeout never call `Tool.execute`,
- skipped calls are returned to the model and do not count as failures,
- true tool failures preserve current stopping behavior,
- parallel approval requests block the whole batch until all resolve.

### Broker and web tests

- initiating-user ownership,
- approve, deny, timeout, conflict, unknown-ID, and expired-ID responses,
- disconnect and exception cleanup,
- approval event ordering before subsequent progress,
- automatic Tier-1 and explicit `tool_agent` callback wiring,
- non-streaming fail-closed behavior.

### Frontend tests

- approval card rendering and sanitised arguments,
- approve and deny submissions,
- button locking and endpoint errors,
- expiry display,
- stream continuation after a decision,
- unchanged behavior for streams without approval events.

The complete backend, frontend, typecheck, Ruff, formatting, and diff checks
must pass.

## Success criteria

1. No side-effecting `ToolAgentLoop` call executes before its per-invocation
   approval.
2. Read-only tool behavior remains unchanged.
3. Missing interactivity, denial, timeout, disconnect, or broker failure never
   executes the side effect.
4. The initiating user can approve or deny from the live frontend without
   restarting the agent run.
5. Parallel generated calls cannot execute partially while approvals remain
   unresolved.
6. Raw or secret arguments are not exposed through SSE, logs, or errors.
7. Existing direct search, search-agent, non-tool chat, and control-flow trace
   behavior remain compatible.

## Out of scope and Phase 2 boundary

Phase 1 does not provide:

- persistence across process restart,
- reconnectable approval streams,
- durable or exactly-once side-effect execution,
- remembered approvals,
- administrator approval,
- cross-process broker coordination,
- scheduled-task approval,
- external-app egress policy,
- approval for direct registry REST or MCP calls.

Phase 2 may introduce a persisted agent-run state machine, durable invocation
records, reconnectable event delivery, idempotency-key propagation, and
restart-safe resume. That work must be designed separately against the runtime
created here; Phase 1 must not imply those guarantees.
