# Design: running transcript for Chat + Tool Agent

Date: 2026-07-21
Status: Approved (brainstorming)
Deliverable 1 of 3 (follow-ups to PR #448)

## Problem

`ChatView` and `ToolAgentView` each render only the **latest** answer — they
`setAnswer("")` on every submit. Multi-turn context is preserved server-side
(via `session_id`), but the user never sees prior turns, so a conversation
looks like it forgets itself.

## Goal

Both conversational surfaces show a **running transcript** of the session's
turns as they accumulate. The Tool Agent transcript interleaves each assistant
turn's tool-call trace.

Decisions locked during brainstorming:
- Applies to **Chat + Tool Agent** (both). Search is stateless (unchanged).
- Client-side accumulation only — no fetch of prior history on mount. The
  transcript grows as the user chats within the session.

## Non-goals (YAGNI)

- No loading of historical turns from the backend on mount.
- No transcript persistence beyond the existing session storage.
- No changes to the backend or the SSE protocol.
- No polish CSS here — that is Deliverable 3. This deliverable adds semantic
  class names only.

## Architecture

Introduce a shared transcript primitive so both views stay DRY and each turn is
rendered one way.

### Types (`web/src/types.ts`)

```typescript
export interface ConversationTurn {
  role: "user" | "assistant";
  content: string;
  toolCalls?: ToolCallTraceView[]; // assistant turns on the Tool Agent surface
  progress?: string[];             // live per-turn progress (Tool Agent)
  pending?: boolean;               // assistant turn still streaming
}
```

### Component (`web/src/components/Transcript.tsx`)

`Transcript({ turns }: { turns: ConversationTurn[] })` renders the list:
- user turn → a user bubble (`turn turn--user`).
- assistant turn → an assistant bubble (`turn turn--assistant`); if
  `toolCalls?.length`, render `<ToolCallTracePanel calls={turn.toolCalls} />`
  above/with the answer; if `progress?.length` and `pending`, render the
  progress list.
- Empty `turns` → renders nothing.

### ChatView

Replace `answer` state with `turns: ConversationTurn[]`. On submit:
1. Append `{ role: "user", content: text }` and an empty
   `{ role: "assistant", content: "", pending: true }`.
2. Stream: `answer` → set the last assistant turn's `content`; `done` → set
   session id + clear `pending`; `error` → surface the error and clear pending.
3. Render `<Transcript turns={turns} />`. Keep the no-model banner + composer.

### ToolAgentView

Same pattern, but the working assistant turn also accumulates:
- `progress` → push to the assistant turn's `progress`.
- `tool_call` → push to the assistant turn's `toolCalls`.
- `answer` → set content; `done` → clear `pending`.

The existing top-level `progress`/`toolCalls`/`answer` states are replaced by
per-turn fields on the transcript.

## Error handling

- Stream error / thrown error: mark the in-flight assistant turn not-pending,
  show the error banner (unchanged). The no-model (`NO_LOCAL_MODEL`) banner is
  unchanged.
- A failed turn keeps the user message in the transcript (so the user sees what
  they asked); the assistant turn shows whatever streamed before the error.

## Testing

Frontend (`web/src/components/__tests__/`):
- `Transcript` renders multiple user/assistant turns in order; an assistant turn
  with `toolCalls` renders the trace.
- `ChatView`: two sequential submits produce a 4-turn transcript (both prior and
  new turns visible) — proves it no longer clears.
- `ToolAgentView`: a turn with streamed `tool_call` shows the trace inside that
  assistant turn; the prior turn remains visible after a second submit.

## Files touched

New:
- `web/src/components/Transcript.tsx`
- `web/src/components/__tests__/Transcript.test.tsx`

Modified:
- `web/src/types.ts` — `ConversationTurn`.
- `web/src/components/ChatView.tsx`, `web/src/components/ToolAgentView.tsx`.
- existing `ChatView.test.tsx` / `ToolAgentView.test.tsx` — updated for the
  transcript (prior turns persist).
