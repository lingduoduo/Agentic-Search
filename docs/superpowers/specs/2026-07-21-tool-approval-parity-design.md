# Design: tool-approval parity for the Tool Agent surface

Date: 2026-07-21
Status: Approved (brainstorming)
Deliverable 2 of 3 (follow-ups to PR #448)

## Problem

`/tool/send-tool-message` runs `_run_tool_agent` with `on_approval=None`
(deferred since #447), so a gated tool executes without prompting. The
auto-router surface (`/api/agent/stream`) already gates tools: it emits
`approval_required` SSE events and waits for the user's decision via the
`ToolApprovalBroker` + `POST /api/agent/approvals/{approval_id}`, rendering a
`ToolApprovalCard`. The Tool Agent tab should have the same behavior.

## Goal

Full parity: on the Tool Agent surface, a gated tool prompts the user before it
runs. Reuse the existing broker, decision endpoint, and card — no new endpoint.

Decisions locked during brainstorming:
- Full parity (interactive gating), not backend-only auto-approve.
- Reuse `_request_tool_approval`, `ToolApprovalBroker`, the decision endpoint
  `POST /api/agent/approvals/{approval_id}`, `submitToolApproval`, and
  `ToolApprovalCard`.

## Non-goals (YAGNI)

- No new approval endpoint (the existing one is generic over the broker).
- No approvals on the non-streaming path (`stream:false`) — approval is
  interactive, so it applies to the SSE path only; non-stream stays
  `on_approval=None`.
- No change to which tools are gated (that policy lives in `ToolAgentLoop`).

## Architecture

### Backend (`tool_backend.py`, streaming path only)

The streaming generator already owns an `asyncio.Queue` and drains it while the
`_run_tool_agent` task runs. Wire approvals into that same queue:

1. Resolve `user` (already done for `user_id`) and
   `broker = getattr(http_request.app.state, "tool_approval_broker", None)`.
2. When `user` is authenticated AND `broker` is not None, build:
   ```python
   async def on_approval(approval_request):
       return await _request_tool_approval(broker, user.id, approval_request, queue)
   ```
   `_request_tool_approval` (module-level in `app.py`) puts
   `{"type":"approval_required","approval":asdict(view)}` on the queue and awaits
   the broker decision, which it returns to the loop. Import it
   **function-locally** (same as `NO_LOCAL_MODEL_MESSAGE`) to stay
   circular-import safe.
3. Pass `on_approval` into `_run` → `_run_tool_agent(..., on_approval=on_approval)`.
   When no user/broker, pass `None` (unchanged behavior).

The drain loop already yields queue items as SSE `data:` lines, so
`approval_required` is delivered to the client with no new plumbing. The client
posts its decision to the existing `POST /api/agent/approvals/{approval_id}`,
the broker resolves, `on_approval` returns, and the loop continues.

Non-streaming path: unchanged (`on_approval=None`).

### Frontend

- `ToolStreamEvent` (the `sendToolMessage` union) gains an `approval_required`
  variant carrying `ToolApprovalView` (both already defined in `types.ts`).
- `ToolAgentView` adds `pendingApprovals: ToolApprovalView[]` state:
  - on `approval_required` → append `e.approval`.
  - render a `ToolApprovalCard` per pending approval (below the transcript,
    above the composer), `onDecision={(d) => submitToolApproval(a.id, d).then(...)}`
    which removes that approval from the list.
  - on `done`/`error`/stream-end → clear `pendingApprovals` (any undecided
    prompts are moot once the turn ends).
- Reuse `submitToolApproval` and `ToolApprovalCard` verbatim.

## Error handling

- No authenticated user, or no broker configured → `on_approval=None`; tools run
  without gating (same as today). Not an error.
- Decision POST fails → `ToolApprovalCard` already surfaces its own error; the
  broker's expiry eventually unblocks the loop.
- Stream error mid-approval → the in-flight turn is marked not-pending and
  pending approvals are cleared.

## Testing

Backend (`tests/unit/test_tool_backend.py`):
- Streaming path emits an `approval_required` event when the loop calls
  `on_approval`: monkeypatch `_run_tool_agent` with a fake that invokes the
  passed `on_approval` with a fake request and asserts a broker is used. Verify
  an `approval_required` SSE event appears and the fake decision flows back.
  (Inject a fake broker on `app.state` whose `request` resolves immediately.)
- When no broker / anonymous user, `_run_tool_agent` receives `on_approval=None`
  (assert via the monkeypatch capturing the kwarg).

Frontend (`web/src/components/__tests__/ToolAgentView.test.tsx`):
- On an `approval_required` event, a `ToolApprovalCard` renders; clicking
  approve calls `submitToolApproval(id, "approve")` and removes the card.

## Files touched

Modified:
- `src/internal/servers/query_and_chat/tool_backend.py` — wire `on_approval`.
- `web/src/types.ts` — add the `approval_required` variant to `ToolStreamEvent`.
- `web/src/components/ToolAgentView.tsx` — pending approvals + card rendering.
- tests (backend + frontend).
- `docs/tool-engine.md` — note that the Tool Agent surface gates tools.
