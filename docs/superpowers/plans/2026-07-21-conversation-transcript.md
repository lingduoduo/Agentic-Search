# Conversation Transcript Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Chat and Tool Agent surfaces render a running transcript of the session's turns instead of only the latest answer.

**Architecture:** A shared `Transcript` component renders `ConversationTurn[]`; both views accumulate turns client-side as the SSE stream arrives.

**Tech Stack:** React 19, TypeScript, Vitest, React Testing Library.

## Global Constraints

- Branch `feat/chat-transcript` (off current `main`; do not commit to `main`).
- Frontend only — no backend/SSE-protocol changes.
- Client-side accumulation only (no history fetch on mount).
- Semantic class names only (`transcript`, `turn`, `turn--user`, `turn--assistant`); no polish CSS (that is a later deliverable).
- Run vitest from `web/`; `npm run typecheck` must stay clean.

---

## Task 1: `ConversationTurn` type + `Transcript` component

**Files:**
- Modify: `web/src/types.ts`
- Create: `web/src/components/Transcript.tsx`
- Test: `web/src/components/__tests__/Transcript.test.tsx`

**Interfaces:**
- Produces: `ConversationTurn` type; `Transcript({ turns }: { turns: ConversationTurn[] })`.

- [ ] **Step 1: Add the type**

Append to `web/src/types.ts` (reuses the existing `ToolCallTraceView`):

```typescript
export interface ConversationTurn {
  role: "user" | "assistant";
  content: string;
  toolCalls?: ToolCallTraceView[];
  progress?: string[];
  pending?: boolean;
}
```

- [ ] **Step 2: Write the failing Transcript test**

Create `web/src/components/__tests__/Transcript.test.tsx`:

```typescript
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Transcript } from "../Transcript";
import type { ConversationTurn } from "../../types";

describe("Transcript", () => {
  it("renders user and assistant turns in order", () => {
    const turns: ConversationTurn[] = [
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ];
    render(<Transcript turns={turns} />);
    expect(screen.getByText("hi")).toBeInTheDocument();
    expect(screen.getByText("hello")).toBeInTheDocument();
  });

  it("renders an assistant turn's tool calls", () => {
    const turns: ConversationTurn[] = [
      {
        role: "assistant",
        content: "done",
        toolCalls: [
          { tool_name: "search", status: "completed", arguments: {}, result_summary: "3 items", latency_ms: 10, error: null },
        ],
      },
    ];
    render(<Transcript turns={turns} />);
    expect(screen.getByText(/search/)).toBeInTheDocument();
  });

  it("renders nothing for an empty transcript", () => {
    const { container } = render(<Transcript turns={[]} />);
    expect(container.querySelector(".transcript")?.children.length ?? 0).toBe(0);
  });
});
```

- [ ] **Step 3: Run to verify failure**

Run: `cd web && npx vitest run src/components/__tests__/Transcript.test.tsx`
Expected: FAIL — cannot resolve `../Transcript`.

- [ ] **Step 4: Implement Transcript**

Create `web/src/components/Transcript.tsx`:

```typescript
import type { ConversationTurn } from "../types";
import { ToolCallTracePanel } from "./ToolCallTracePanel";

export function Transcript({ turns }: { turns: ConversationTurn[] }) {
  return (
    <div className="transcript">
      {turns.map((turn, i) => (
        <div key={i} className={`turn turn--${turn.role}`}>
          {turn.role === "assistant" &&
            turn.progress &&
            turn.pending &&
            turn.progress.length > 0 && (
              <ul className="turn__progress">
                {turn.progress.map((p, j) => (
                  <li key={j}>{p}</li>
                ))}
              </ul>
            )}
          {turn.toolCalls && turn.toolCalls.length > 0 && (
            <ToolCallTracePanel calls={turn.toolCalls} />
          )}
          {turn.content && <div className="turn__content">{turn.content}</div>}
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 5: Run to green + typecheck**

Run: `cd web && npx vitest run src/components/__tests__/Transcript.test.tsx && npm run typecheck`
Expected: 3 pass; typecheck clean.

- [ ] **Step 6: Commit**

```bash
git add web/src/types.ts web/src/components/Transcript.tsx web/src/components/__tests__/Transcript.test.tsx
git commit -m "feat(web): ConversationTurn type + shared Transcript component"
```

---

## Task 2: Wire Transcript into ChatView + ToolAgentView

**Files:**
- Modify: `web/src/components/ChatView.tsx`, `web/src/components/ToolAgentView.tsx`
- Test: update `web/src/components/__tests__/ChatView.test.tsx`, `ToolAgentView.test.tsx`

**Interfaces:**
- Consumes: `Transcript`, `ConversationTurn` (Task 1); existing `sendChatMessage`/`sendToolMessage`.

- [ ] **Step 1: Refactor ChatView to a transcript**

Replace `web/src/components/ChatView.tsx` with:

```typescript
import { useState } from "react";
import { sendChatMessage } from "../api";
import type { ConversationTurn } from "../types";
import { Transcript } from "./Transcript";

export function ChatView() {
  const [message, setMessage] = useState("");
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [noModel, setNoModel] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function patchLastAssistant(patch: Partial<ConversationTurn>) {
    setTurns((prev) => {
      const next = [...prev];
      for (let i = next.length - 1; i >= 0; i--) {
        if (next[i].role === "assistant") {
          next[i] = { ...next[i], ...patch };
          break;
        }
      }
      return next;
    });
  }

  async function submit() {
    const text = message.trim();
    if (!text || busy) return;
    setBusy(true); setError(null); setNoModel(false);
    setTurns((prev) => [
      ...prev,
      { role: "user", content: text },
      { role: "assistant", content: "", pending: true },
    ]);
    try {
      for await (const e of sendChatMessage({ message: text, session_id: sessionId })) {
        if (e.type === "answer") patchLastAssistant({ content: e.text });
        else if (e.type === "done") { setSessionId(e.session_id); patchLastAssistant({ pending: false }); }
        else if (e.type === "error") { setError(e.detail); patchLastAssistant({ pending: false }); }
      }
      setMessage("");
    } catch (err) {
      patchLastAssistant({ pending: false });
      if (err instanceof Error && err.message === "NO_LOCAL_MODEL") setNoModel(true);
      else setError(err instanceof Error ? err.message : "Chat failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="chat-view" aria-label="Chat">
      {noModel && (
        <div className="error-banner" role="alert">
          Chat needs a local model — set <code>SEARCH_AGENT_MODEL</code> (or{" "}
          <code>SEARCH_AGENT_SERVER_URL</code>) in <code>.env</code> and restart the backend.
        </div>
      )}
      <Transcript turns={turns} />
      {error && <div className="error-banner">{error}</div>}
      <div className="chat-view__composer">
        <input
          aria-label="Chat message"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder="Message the model directly…"
          disabled={busy}
        />
        <button onClick={submit} disabled={busy}>{busy ? "…" : "Send"}</button>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Refactor ToolAgentView to a transcript**

Replace `web/src/components/ToolAgentView.tsx` with:

```typescript
import { useState } from "react";
import { sendToolMessage } from "../api";
import type { ConversationTurn } from "../types";
import { Transcript } from "./Transcript";

export function ToolAgentView() {
  const [message, setMessage] = useState("");
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [noModel, setNoModel] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function patchLastAssistant(fn: (t: ConversationTurn) => ConversationTurn) {
    setTurns((prev) => {
      const next = [...prev];
      for (let i = next.length - 1; i >= 0; i--) {
        if (next[i].role === "assistant") { next[i] = fn(next[i]); break; }
      }
      return next;
    });
  }

  async function submit() {
    const text = message.trim();
    if (!text || busy) return;
    setBusy(true); setError(null); setNoModel(false);
    setTurns((prev) => [
      ...prev,
      { role: "user", content: text },
      { role: "assistant", content: "", toolCalls: [], progress: [], pending: true },
    ]);
    try {
      for await (const e of sendToolMessage({ message: text, session_id: sessionId })) {
        if (e.type === "progress")
          patchLastAssistant((t) => ({ ...t, progress: [...(t.progress ?? []), e.text] }));
        else if (e.type === "tool_call")
          patchLastAssistant((t) => ({ ...t, toolCalls: [...(t.toolCalls ?? []), e] }));
        else if (e.type === "answer")
          patchLastAssistant((t) => ({ ...t, content: e.text }));
        else if (e.type === "done") {
          setSessionId(e.session_id);
          patchLastAssistant((t) => ({ ...t, pending: false }));
        } else if (e.type === "error") {
          setError(e.detail);
          patchLastAssistant((t) => ({ ...t, pending: false }));
        }
      }
      setMessage("");
    } catch (err) {
      patchLastAssistant((t) => ({ ...t, pending: false }));
      if (err instanceof Error && err.message === "NO_LOCAL_MODEL") setNoModel(true);
      else setError(err instanceof Error ? err.message : "Tool agent failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="tool-agent-view" aria-label="Tool Agent">
      {noModel && (
        <div className="error-banner" role="alert">
          Tool Agent needs a local model — set <code>SEARCH_AGENT_MODEL</code> (or{" "}
          <code>SEARCH_AGENT_SERVER_URL</code>) in <code>.env</code> and restart the backend.
        </div>
      )}
      <Transcript turns={turns} />
      {error && <div className="error-banner">{error}</div>}
      <div className="tool-agent-view__composer">
        <input
          aria-label="Tool agent message"
          value={message}
          onChange={(ev) => setMessage(ev.target.value)}
          onKeyDown={(ev) => ev.key === "Enter" && submit()}
          placeholder="Ask the tool agent to do something…"
          disabled={busy}
        />
        <button onClick={submit} disabled={busy}>{busy ? "Running…" : "Send"}</button>
      </div>
    </section>
  );
}
```

- [ ] **Step 3: Update the view tests for the transcript**

The existing `ChatView.test.tsx` / `ToolAgentView.test.tsx` assert on the single-answer render; they should still pass (the streamed answer is now inside a turn, still found by `getByText`). ADD one persistence test to each. Append to `ChatView.test.tsx`:

```typescript
it("keeps prior turns visible across two submits", async () => {
  const answers = ["first answer", "second answer"];
  let call = 0;
  vi.spyOn(api, "sendChatMessage").mockImplementation((() => {
    const text = answers[call++];
    async function* g() {
      yield { type: "answer", text } as const;
      yield { type: "done", session_id: "s1" } as const;
    }
    return g();
  }) as never);

  render(<ChatView />);
  const input = screen.getByLabelText("Chat message");
  fireEvent.change(input, { target: { value: "q1" } });
  fireEvent.click(screen.getByText("Send"));
  await waitFor(() => expect(screen.getByText("first answer")).toBeInTheDocument());
  fireEvent.change(input, { target: { value: "q2" } });
  fireEvent.click(screen.getByText("Send"));
  await waitFor(() => expect(screen.getByText("second answer")).toBeInTheDocument());
  // prior turn still present
  expect(screen.getByText("first answer")).toBeInTheDocument();
  expect(screen.getByText("q1")).toBeInTheDocument();
});
```

Append the analogous persistence test to `ToolAgentView.test.tsx` (using `sendToolMessage`, asserting the first turn's answer remains after a second submit). If either file's existing "no-model banner" test asserted the exact prior markup, adjust only what the transcript changed (the answer text is now inside `.turn__content` but still matched by `getByText`).

- [ ] **Step 4: Run the full frontend suite + typecheck**

Run: `cd web && npm run typecheck && npx vitest run`
Expected: no type errors; all tests pass (Transcript + updated view tests + the untouched suite).

- [ ] **Step 5: Commit**

```bash
git add web/src/components/ChatView.tsx web/src/components/ToolAgentView.tsx web/src/components/__tests__/ChatView.test.tsx web/src/components/__tests__/ToolAgentView.test.tsx
git commit -m "feat(web): Chat + Tool Agent render a running transcript"
```

---

## Self-Review

- **Spec coverage:** shared `Transcript` + `ConversationTurn` (Task 1) ✓; Chat transcript (Task 2) ✓; Tool Agent transcript with interleaved tool traces + per-turn progress (Task 2) ✓; client-side accumulation, no history fetch ✓; prior-turns-persist tests ✓; semantic classes only, no polish CSS ✓.
- **Placeholder scan:** none; all steps have runnable code + commands.
- **Type consistency:** `ConversationTurn` fields used in Transcript match the type; `patchLastAssistant` mutates the last assistant turn consistently in both views; `toolCalls` reuses `ToolCallTraceView`.
