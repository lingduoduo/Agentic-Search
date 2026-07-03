# Drop the chat-column control-flow trace panel — design

**Date:** 2026-07-03
**Status:** Approved (design).
**Scope:** Frontend only (`web/`). Remove the redundant fine-grained
control-flow trace from the user-facing chat answer column; the detailed view
already lives in the Dev Console.

## Problem

The chat answer column renders two "what the agent did" surfaces stacked
together:

1. `AnswerPanel.ProgressLog` — the coarse, user-facing live progress feed
   ("Agent reasoning · Turn N · …"). This is the named streaming-SSE-progress
   feature and stays.
2. `ControlFlowTracePanel` — a fine-grained per-event list (planner /
   loop_controller / search_tool / … with status, duration, doc counts).

`ControlFlowTracePanel` is fed from the same `controlFlowTrace` state that the
Dev Console already renders through `RequestTracePanel` (a timeline/gantt view).
So the detailed trace is shown **twice** — once cluttering the chat, once in the
Dev Console — and it also duplicates the *purpose* of the progress log directly
above it. It is a developer-observability surface that does not belong in the
user-facing chat.

## Change

Surgical, frontend-only:

1. **`web/src/App.tsx`** — remove the `ControlFlowTracePanel` import and its
   `<ControlFlowTracePanel events={controlFlowTrace} live={isLoading} />` render
   in the answer column. **Keep** the `controlFlowTrace` state, the `upsertTrace`
   helper, the SSE `trace`/`done` handlers, and the
   `<DevConsole controlFlowTrace={…}>` wiring — the trace still flows to the Dev
   Console unchanged.
2. **Delete** `web/src/components/ControlFlowTracePanel.tsx` and
   `web/src/components/__tests__/ControlFlowTracePanel.test.tsx`.
3. **`web/src/components/__tests__/App.test.tsx`** — remove the single test
   asserting the "show control flow" summary button in the chat column. The
   detailed trace rendering is covered by
   `web/src/components/__tests__/RequestTracePanel.test.tsx`.
4. **`web/src/styles.css`** — remove the now-dead `.control-flow-*` rules
   (owned solely by the deleted component; `RequestTracePanel` uses its own
   `.request-trace-*` classes).

## Behavior after

Chat answer column: `ProgressLog → IntentBadge → Answer (markdown + [D1] links)
→ [tool intent] ToolCallTracePanel`. No user-visible data loss — full
control-flow detail remains available in the Dev Console.

## Non-goals

- No change to the Dev Console, `RequestTracePanel`, or the SSE protocol.
- No change to `ControlFlowEventView` / `controlFlowTrace` state (still consumed
  by the Dev Console).
- `ToolCallTracePanel`, `SessionTimeline`, `IntentBadge`, and the route pill are
  untouched (considered non-redundant this pass).

## Testing

- `cd web && npm run typecheck` — clean (no dangling imports/types).
- `cd web && npm test` — green after removing the two test files' worth of
  ControlFlowTracePanel/App-chat-trace assertions; all other component tests
  unchanged.

## Files touched

- **Modify:** `web/src/App.tsx`, `web/src/components/__tests__/App.test.tsx`,
  `web/src/styles.css`
- **Delete:** `web/src/components/ControlFlowTracePanel.tsx`,
  `web/src/components/__tests__/ControlFlowTracePanel.test.tsx`
