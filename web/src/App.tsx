import { FormEvent, useMemo, useState } from "react";
import { Bot, FileSearch, MessageSquarePlus, Search } from "lucide-react";
import { createSession, runAgent } from "./api";
import { AnswerPanel } from "./components/AnswerPanel";
import { SearchComposer } from "./components/SearchComposer";
import { SessionTimeline } from "./components/SessionTimeline";
import { SourceGrid } from "./components/SourceGrid";
import type {
  AgentExperienceResponse,
  ChatMessageView,
  SourceDocumentView,
} from "./types";

const DEFAULT_SEARCH_URL = "http://localhost:8000/retrieve";

export function App() {
  const [query, setQuery] = useState("");
  const [searchUrl, setSearchUrl] = useState(DEFAULT_SEARCH_URL);
  const [topK, setTopK] = useState(5);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<string[]>([]);
  const [documents, setDocuments] = useState<SourceDocumentView[]>([]);
  const [messages, setMessages] = useState<ChatMessageView[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const status = useMemo(() => {
    if (isLoading) return "Searching";
    if (error) return "Needs attention";
    if (answer) return "Grounded";
    return "Ready";
  }, [answer, error, isLoading]);

  async function ensureSession() {
    if (sessionId) return sessionId;
    const session = await createSession({ title: "Search session" });
    setSessionId(session.id);
    setMessages(session.messages);
    return session.id;
  }

  async function handleSubmit(event?: FormEvent) {
    event?.preventDefault();
    const normalizedQuery = query.trim();
    if (!normalizedQuery) return;
    setIsLoading(true);
    setError(null);
    try {
      const activeSessionId = await ensureSession();
      const response: AgentExperienceResponse = await runAgent({
        query: normalizedQuery,
        session_id: activeSessionId,
        search_url: searchUrl,
        top_k: topK,
      });
      setSessionId(response.session_id);
      setAnswer(response.answer);
      setCitations(response.citations);
      setDocuments(response.documents);
      setMessages(response.messages);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Search failed");
      setDocuments([]);
    } finally {
      setIsLoading(false);
    }
  }

  async function handleNewSession() {
    const session = await createSession({ title: "Search session" });
    setSessionId(session.id);
    setAnswer("");
    setCitations([]);
    setDocuments([]);
    setMessages([]);
    setError(null);
  }

  return (
    <main className="app-shell">
      <section className="workspace" aria-label="Agentic Search workspace">
        <header className="topbar">
          <div className="brand">
            <div className="brand-mark" aria-hidden="true">
              <FileSearch size={26} />
            </div>
            <div>
              <h1>Agentic Search</h1>
              <p>Ask a question, retrieve context, and inspect the answer trail.</p>
            </div>
          </div>
          <div className="topbar-actions">
            <span className="status-pill">{status}</span>
            <button className="icon-button" type="button" onClick={handleNewSession}>
              <MessageSquarePlus size={18} />
              <span>New</span>
            </button>
          </div>
        </header>

        <SearchComposer
          query={query}
          searchUrl={searchUrl}
          topK={topK}
          isLoading={isLoading}
          onQueryChange={setQuery}
          onSearchUrlChange={setSearchUrl}
          onTopKChange={setTopK}
          onSubmit={handleSubmit}
        />

        {error && <div className="error-banner">{error}</div>}

        <div className="content-grid">
          <section className="answer-column" aria-label="Answer">
            <div className="section-heading">
              <Bot size={18} />
              <h2>Answer</h2>
            </div>
            <AnswerPanel answer={answer} citations={citations} />
          </section>

          <aside className="side-column" aria-label="Sources and session">
            <section className="panel">
              <div className="section-heading">
                <Search size={18} />
                <h2>Sources</h2>
                <span className="count">{documents.length}</span>
              </div>
              <SourceGrid documents={documents} />
            </section>

            <section className="panel">
              <div className="section-heading">
                <MessageSquarePlus size={18} />
                <h2>Session</h2>
              </div>
              <SessionTimeline messages={messages} />
            </section>
          </aside>
        </div>
      </section>
    </main>
  );
}
