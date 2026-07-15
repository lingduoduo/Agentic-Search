# Generated Context Pack

# Drop Chat Control Flow Panel

## Sources

- [Specification: 2026-07-03-drop-chat-control-flow-panel-design.md](../specs/2026-07-03-drop-chat-control-flow-panel-design.md)
- [Plan: 2026-07-03-drop-chat-control-flow-panel.md](../plans/2026-07-03-drop-chat-control-flow-panel.md)

## Specification Context

### Non-goals

- No change to the Dev Console, `RequestTracePanel`, or the SSE protocol.
- No change to `ControlFlowEventView` / `controlFlowTrace` state (still consumed
  by the Dev Console).
- `ToolCallTracePanel`, `SessionTimeline`, `IntentBadge`, and the route pill are
  untouched (considered non-redundant this pass).

### Testing

- `cd web && npm run typecheck` — clean (no dangling imports/types).
- `cd web && npm test` — green after removing the two test files' worth of
  ControlFlowTracePanel/App-chat-trace assertions; all other component tests
  unchanged.

## Implementation Plan Context

### Global Constraints

- **Frontend only.** No backend / SSE protocol change. `ControlFlowEventView` and `controlFlowTrace` state stay (Dev Console consumes them).
- **No user-visible data loss.** Full control-flow detail remains in the Dev Console via `RequestTracePanel`.
- **Green gates:** `npm run typecheck` and `npm test` pass.

---

### Task 1: Remove the panel from App and delete the component

- [x] **Step 1:** Delete `web/src/components/ControlFlowTracePanel.tsx` + `web/src/components/__tests__/ControlFlowTracePanel.test.tsx`.
- [x] **Step 2:** In `App.tsx`, remove the `ControlFlowTracePanel` import and its render line; verify `controlFlowTrace`, `upsertTrace`, `ControlFlowEventView`, and the `DevConsole` wiring remain.
- [x] **Verify:** `npm run typecheck` clean (no dangling references).

### Task 2: Prune tests and dead CSS

- [x] **Step 1:** Remove the "shows the authoritative control-flow trace after streaming completes" test from `App.test.tsx`.
- [x] **Step 2:** Remove the `.control-flow-*` blocks from `styles.css` (leave `.request-trace-*` intact).
- [x] **Verify:** `npm test` green.

### Task 3: Full verification

- [x] `cd web && npm run typecheck && npm test` both green.
- [x] Grep confirms zero remaining `ControlFlowTracePanel` references.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
