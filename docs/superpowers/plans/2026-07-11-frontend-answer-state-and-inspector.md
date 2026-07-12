# Plan: Frontend answer-state reset, session timeline, and inspector selection

Date: 2026-07-11
Branch: `fix/frontend-answer-state-and-inspector`
Spec: `docs/superpowers/specs/2026-07-11-frontend-answer-state-and-inspector-design.md`

## Steps

1. **F2 — reset answer state on new/failed query** (`web/src/App.tsx`)
   → verify: submit-start resets `answer`/`streamingAnswer`/`citations`/`documents`/`toolCalls`;
     `catch` also clears `answer` + `citations`.

2. **F1 — populate the session timeline** (`web/src/App.tsx`)
   → verify: append user message after `ensureSession`; append assistant message on `done`.

3. **F3 — fix inspector selection precedence** (`web/src/components/debug/RequestInspector.tsx`)
   → verify: detail id is `selected ?? selectedRequestId` so a manual click wins.

4. **Tests** (`web/src/components/__tests__/`)
   → verify: new App tests for F2 + F1; new `RequestInspector.test.tsx` for F3;
     scope existing App answer assertions to the answer column.

5. **Gates**
   → verify: `cd web && npm run typecheck` clean; `cd web && npm run test -- --run` green.

## Result

- typecheck: clean.
- vitest: 127 passed (17 files).
