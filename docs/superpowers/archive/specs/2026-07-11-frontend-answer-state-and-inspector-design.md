# Spec: Frontend answer-state reset, session timeline, and inspector selection

Date: 2026-07-11
Branch: `fix/frontend-answer-state-and-inspector`

Three verified frontend bugs in `web/src/`. All fixes are frontend-only and behavior-preserving apart from the specific broken paths.

## Bug F2 — error path leaves a stale answer + dead citation links

**Symptom.** Run query A (renders answer + `[D1]` citation links), then run query B
that errors. The user sees the red error banner *plus* query A's stale answer and
citation anchors that now point at removed source cards.

**Root cause.** `App.tsx` `handleSubmit`:
- On submit start, `answer`, `streamingAnswer`, `citations`, `documents`, `toolCalls`
  are not reset.
- The `catch` branch clears `documents` but leaves `answer` and `citations`.

**Fix.**
- At the start of a submit (before streaming), reset `answer`, `streamingAnswer`,
  `citations`, `documents`, `toolCalls` (plus the already-reset `intent`/`error`).
- In the `catch` branch also clear `answer` and `citations` (in addition to `documents`).
- The abort guard (`requestRef.current !== controller`) and the successful-stream
  path are unchanged.

## Bug F1 — SessionTimeline never populates in the streaming flow

**Symptom.** The "Session" panel always shows "Start a query to create history."

**Root cause.** `messages` is only set by `ensureSession` (empty for a new session)
and `handleNewSession` (reset). The SSE `done` event carries no `messages`, and no
code calls `setMessages` after a turn.

**Fix (frontend-only).** Build the timeline from local state:
- On submit, append `{ role: "user", content: query }`. Done *after* `ensureSession`
  resolves so the new-session `setMessages([])` inside it cannot wipe the append.
- On `done`, append `{ role: "assistant", content: <final answer> }`.
- `handleNewSession` still clears `messages`.
- `SessionTimeline` keys off the message's index in the full array, which stays
  stable and unique across appends.

## Bug F3 — RequestInspector run-selection permanently stuck (dev-console-gated)

**Symptom.** Clicking a different run in the list never changes the detail pane.

**Root cause.** `RequestInspector.tsx` detail effect: `const id = selectedRequestId ?? selected;`.
`selectedRequestId` (from `App.lastRequestId`) is always truthy after the first
request, so it always wins and the manual `selected` is ignored.

**Fix.** Invert precedence to `const id = selected ?? selectedRequestId;`. Default
`selected` is `null`, so the pane auto-follows the streamed latest request; once the
user clicks a run, `selected` is set and takes priority.

## Testing

- App F2: after a grounded turn then an errored follow-up, the answer column has no
  citation links and no stale answer text.
- App F1: after a streamed turn the session panel shows a user row (query) then an
  assistant row (answer).
- RequestInspector F3: with `selectedRequestId="req-A"`, a user click on `req-B`
  makes the detail pane load `req-B`.

Existing App tests that asserted on the answer via `getByText` were scoped to the
answer column, because the answer now legitimately renders in both the answer panel
and the session timeline.
