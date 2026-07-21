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
        if (next[i].role === "assistant") {
          next[i] = fn(next[i]);
          break;
        }
      }
      return next;
    });
  }

  async function submit() {
    const text = message.trim();
    if (!text || busy) return;
    setBusy(true);
    setError(null);
    setNoModel(false);
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
        <button onClick={submit} disabled={busy}>
          {busy ? "Running…" : "Send"}
        </button>
      </div>
    </section>
  );
}
