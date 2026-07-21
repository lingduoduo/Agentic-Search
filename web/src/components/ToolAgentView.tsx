import { useState } from "react";
import { sendToolMessage } from "../api";
import type { ToolCallTraceView } from "../types";
import { ToolCallTracePanel } from "./ToolCallTracePanel";

export function ToolAgentView() {
  const [message, setMessage] = useState("");
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [answer, setAnswer] = useState("");
  const [progress, setProgress] = useState<string[]>([]);
  const [toolCalls, setToolCalls] = useState<ToolCallTraceView[]>([]);
  const [noModel, setNoModel] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    const text = message.trim();
    if (!text || busy) return;
    setBusy(true);
    setError(null);
    setNoModel(false);
    setAnswer("");
    setProgress([]);
    setToolCalls([]);
    try {
      for await (const e of sendToolMessage({ message: text, session_id: sessionId })) {
        if (e.type === "progress") setProgress((p) => [...p, e.text]);
        else if (e.type === "tool_call") setToolCalls((c) => [...c, e]);
        else if (e.type === "answer") setAnswer(e.text);
        else if (e.type === "done") setSessionId(e.session_id);
        else if (e.type === "error") setError(e.detail);
      }
      setMessage("");
    } catch (err) {
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
      {error && <div className="error-banner">{error}</div>}
      {progress.length > 0 && (
        <ul className="tool-agent-view__progress">
          {progress.map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ul>
      )}
      {toolCalls.length > 0 && <ToolCallTracePanel calls={toolCalls} />}
      {answer && <div className="tool-agent-view__answer">{answer}</div>}
    </section>
  );
}
