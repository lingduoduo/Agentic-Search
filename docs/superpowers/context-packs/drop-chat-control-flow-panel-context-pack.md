# Generated Context Pack

# Drop Chat Control Flow Panel

## Sources

- [Specification: 2026-07-03-drop-chat-control-flow-panel-design.md](../archive/specs/2026-07-03-drop-chat-control-flow-panel-design.md)
- [Plan: 2026-07-03-drop-chat-control-flow-panel.md](../archive/plans/2026-07-03-drop-chat-control-flow-panel.md)

## Specification Context

### Overview

**Date:** 2026-07-03
**Status:** Approved (design).
**Scope:** Frontend only (`web/`). Remove the redundant fine-grained
control-flow trace from the user-facing chat answer column; the detailed view
already lives in the Dev Console.

## Implementation Plan Context

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
