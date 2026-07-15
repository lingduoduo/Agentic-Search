# Drop Chat Control-Flow Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Remove the redundant `ControlFlowTracePanel` from the chat answer column; keep the detailed trace in the Dev Console.

**Architecture:** Frontend-only deletion. `App.tsx` stops rendering the panel but keeps the `controlFlowTrace` state feeding `DevConsole`. The component, its test, the one App-level chat-trace test, and the dead CSS are removed.

**Tech Stack:** React 19 + Vite + TypeScript, vitest. No new deps.

**Spec:** `docs/superpowers/specs/2026-07-03-drop-chat-control-flow-panel-design.md`.

## Global Constraints

- **Frontend only.** No backend / SSE protocol change. `ControlFlowEventView` and `controlFlowTrace` state stay (Dev Console consumes them).
- **No user-visible data loss.** Full control-flow detail remains in the Dev Console via `RequestTracePanel`.
- **Green gates:** `npm run typecheck` and `npm test` pass.

---

## File Structure

- **Modify** `web/src/App.tsx` — drop import + `<ControlFlowTracePanel>` render; keep state/handlers/DevConsole.
- **Modify** `web/src/components/__tests__/App.test.tsx` — remove the "show control flow" chat-column test.
- **Modify** `web/src/styles.css` — remove dead `.control-flow-*` rules.
- **Delete** `web/src/components/ControlFlowTracePanel.tsx`, `web/src/components/__tests__/ControlFlowTracePanel.test.tsx`.

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
