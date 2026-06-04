import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import { Bot, FileSearch, MessageSquarePlus, Search } from "lucide-react";
import { createSession, getAdminSummary, runAgent } from "./api";
import { AdminOverview } from "./components/AdminOverview";
import { AnswerPanel } from "./components/AnswerPanel";
import { SearchComposer } from "./components/SearchComposer";
import { SessionTimeline } from "./components/SessionTimeline";
import { SourceGrid } from "./components/SourceGrid";
import type {
  AdminSurfaceSummary,
  AgentExperienceResponse,
  AgentMode,
  ChatMessageView,
  SourceDocumentView,
} from "./types";

const DEFAULT_SEARCH_URL = "http://localhost:8000/retrieve";

export function App() {
  const [query, setQuery] = useState("");
  const [searchUrl, setSearchUrl] = useState(DEFAULT_SEARCH_URL);
  const [topK, setTopK] = useState(5);
  const [mode, setMode] = useState<AgentMode>("chat_once");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<string[]>([]);
  const [documents, setDocuments] = useState<SourceDocumentView[]>([]);
  const [messages, setMessages] = useState<ChatMessageView[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [adminSummary, setAdminSummary] = useState<AdminSurfaceSummary | null>(null);
  const requestRef = useRef<AbortController | null>(null);

  useEffect(() => {
    getAdminSummary().then(setAdminSummary).catch(() => undefined);
  }, []);

  const status = useMemo(() => {
    if (isLoading) return "Searching";
    if (error) return "Needs attention";
    if (answer) return "Grounded";
    return "Ready";
  }, [answer, error, isLoading]);

  const ensureSession = useCallback(
    async (signal: AbortSignal) => {
      if (sessionId) return sessionId;
      const session = await createSession({ title: "Search session" }, { signal });
      setSessionId(session.id);
      setMessages(session.messages);
      return session.id;
    },
    [sessionId],
  );

  const handleSubmit = useCallback(async (event?: FormEvent) => {
    event?.preventDefault();
    const normalizedQuery = query.trim();
    if (!normalizedQuery) return;

    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;

    setIsLoading(true);
    setError(null);
    try {
      const activeSessionId = await ensureSession(controller.signal);
      const response: AgentExperienceResponse = await runAgent({
        query: normalizedQuery,
        session_id: activeSessionId,
        search_url: searchUrl,
        top_k: topK,
        mode,
      }, { signal: controller.signal });
      setSessionId(response.session_id);
      setAnswer(response.answer);
      setCitations(response.citations);
      setDocuments(response.documents);
      setMessages(response.messages);
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setError(caught instanceof Error ? caught.message : "Search failed");
      setDocuments([]);
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setIsLoading(false);
      }
    }
  }, [ensureSession, mode, query, searchUrl, topK]);

  const handleNewSession = useCallback(async () => {
    requestRef.current?.abort();
    const session = await createSession({ title: "Search session" });
    setSessionId(session.id);
    setAnswer("");
    setCitations([]);
    setDocuments([]);
    setMessages([]);
    setError(null);
    setIsLoading(false);
  }, []);

  const handleTopKChange = useCallback((value: number) => {
    setTopK(Math.min(20, Math.max(1, value || 1)));
  }, []);

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
          mode={mode}
          isLoading={isLoading}
          onQueryChange={setQuery}
          onSearchUrlChange={setSearchUrl}
          onTopKChange={handleTopKChange}
          onModeChange={setMode}
          onSubmit={handleSubmit}
        />

        {error && <div className="error-banner">{error}</div>}

        {adminSummary && <AdminOverview summary={adminSummary} />}

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
