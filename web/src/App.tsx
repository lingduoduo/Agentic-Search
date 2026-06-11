import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import { Bot, FileSearch, MessageSquarePlus, Search } from "lucide-react";
import {
  createSession,
  getAdminSummary,
  getAnalyticsByFlow,
  getAnalyticsByLLM,
  getAnalyticsByPersona,
  runAgent,
} from "./api";
import { AdminOverview } from "./components/AdminOverview";
import { AnalyticsDashboard } from "./components/AnalyticsDashboard";
import { AnswerPanel } from "./components/AnswerPanel";
import { SearchComposer } from "./components/SearchComposer";
import { SessionTimeline } from "./components/SessionTimeline";
import { SourceGrid } from "./components/SourceGrid";
import type {
  AdminSurfaceSummary,
  AgentExperienceResponse,
  AgentMode,
  BreakdownAnalytics,
  ChatMessageView,
  SearchSourceProvider,
  SourceDocumentView,
} from "./types";

const DEFAULT_SEARCH_URL = "http://localhost:8000/retrieve";
const DEFAULT_BROWSER_SEARCH_URL = "http://localhost:8001/retrieve";

export function App() {
  const [query, setQuery] = useState("");
  const [searchUrl, setSearchUrl] = useState(DEFAULT_SEARCH_URL);
  const [topK, setTopK] = useState(5);
  const [mode, setMode] = useState<AgentMode>("chat_once");
  const [sourceProvider, setSourceProvider] =
    useState<SearchSourceProvider>("retrieval");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<string[]>([]);
  const [documents, setDocuments] = useState<SourceDocumentView[]>([]);
  const [messages, setMessages] = useState<ChatMessageView[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [adminSummary, setAdminSummary] = useState<AdminSurfaceSummary | null>(null);
  const [analyticsByLLM, setAnalyticsByLLM] = useState<BreakdownAnalytics | null>(null);
  const [analyticsByPersona, setAnalyticsByPersona] = useState<BreakdownAnalytics | null>(null);
  const [analyticsByFlow, setAnalyticsByFlow] = useState<BreakdownAnalytics | null>(null);
  const requestRef = useRef<AbortController | null>(null);

  useEffect(() => {
    getAdminSummary().then(setAdminSummary).catch(() => undefined);
    getAnalyticsByLLM().then(setAnalyticsByLLM).catch(() => undefined);
    getAnalyticsByPersona().then(setAnalyticsByPersona).catch(() => undefined);
    getAnalyticsByFlow().then(setAnalyticsByFlow).catch(() => undefined);
  }, []);

  const status = useMemo(() => {
    if (isLoading) return "Searching";
    if (error) return "Needs attention";
    if (answer) return "Grounded";
    return "Ready";
  }, [answer, error, isLoading]);
  const isChatMode = mode === "chat_once" || mode === "chat_loop";
  const isSearchMode = mode === "search_tool" || mode === "hybrid_search";

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
        source_provider: isSearchMode ? sourceProvider : "retrieval",
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
  }, [ensureSession, isSearchMode, mode, query, searchUrl, sourceProvider, topK]);

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

  const handleSourceProviderChange = useCallback((value: SearchSourceProvider) => {
    setSourceProvider(value);
    setSearchUrl((current) => {
      if (value === "browser" && current === DEFAULT_SEARCH_URL) {
        return DEFAULT_BROWSER_SEARCH_URL;
      }
      if (
        (value === "retrieval" || value === "all") &&
        current === DEFAULT_BROWSER_SEARCH_URL
      ) {
        return DEFAULT_SEARCH_URL;
      }
      return current;
    });
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
          sourceProvider={sourceProvider}
          isLoading={isLoading}
          onQueryChange={setQuery}
          onSearchUrlChange={setSearchUrl}
          onTopKChange={handleTopKChange}
          onModeChange={setMode}
          onSourceProviderChange={handleSourceProviderChange}
          onSubmit={handleSubmit}
        />

        {error && <div className="error-banner">{error}</div>}

        {adminSummary && <AdminOverview summary={adminSummary} />}
        {(analyticsByLLM || analyticsByPersona || analyticsByFlow) && (
          <AnalyticsDashboard
            byLLM={analyticsByLLM}
            byPersona={analyticsByPersona}
            byFlow={analyticsByFlow}
          />
        )}

        <div className="results-layout">
          <section className="answer-column" aria-label={isChatMode ? "Answer" : "Search Summary"}>
            <div className="section-heading">
              {isChatMode ? <Bot size={18} /> : <Search size={18} />}
              <h2>{isChatMode ? "Answer" : "Search Summary"}</h2>
            </div>
            <AnswerPanel answer={answer} citations={citations} />
          </section>

          <section className="panel sources-panel wide" aria-label="Sources">
            <div className="section-heading">
              <Search size={18} />
              <h2>Sources</h2>
              <span className="count">{documents.length}</span>
            </div>
            <SourceGrid documents={documents} />
          </section>

          <section className="panel" aria-label="Session">
            <div className="section-heading">
              <MessageSquarePlus size={18} />
              <h2>Session</h2>
            </div>
            <SessionTimeline messages={messages} />
          </section>
        </div>
      </section>
    </main>
  );
}
