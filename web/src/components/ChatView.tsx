import { useState } from "react";
import { sendChatMessage } from "../api";

export function ChatView() {
  const [message, setMessage] = useState("");
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [answer, setAnswer] = useState("");
  const [noModel, setNoModel] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    const text = message.trim();
    if (!text || busy) return;
    setBusy(true); setError(null); setNoModel(false); setAnswer("");
    try {
      for await (const e of sendChatMessage({ message: text, session_id: sessionId })) {
        if (e.type === "answer") setAnswer(e.text);
        else if (e.type === "done") setSessionId(e.session_id);
        else if (e.type === "error") setError(e.detail);
      }
      setMessage("");
    } catch (err) {
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
      {error && <div className="error-banner">{error}</div>}
      {answer && <div className="chat-view__answer">{answer}</div>}
    </section>
  );
}
