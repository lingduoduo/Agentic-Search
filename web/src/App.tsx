import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import { Bot, ClipboardList, FileSearch, MessageSquarePlus, Plug, Search, Wrench } from "lucide-react";
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
import { ConnectorPanel } from "./components/ConnectorPanel";
import { QueryHistoryPanel } from "./components/QueryHistoryPanel";
import { ToolPanel } from "./components/ToolPanel";
import { SearchComposer } from "./components/SearchComposer";
import { SessionTimeline } from "./components/SessionTimeline";
import { SourceGrid } from "./components/SourceGrid";
import type {
  AdminSurfaceSummary,
  AgentExperienceRequest,
  AgentExperienceResponse,
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
  const [intent, setIntent] = useState<"search" | "chat" | "tool" | undefined>(undefined);
  const [sourceProvider, setSourceProvider] =
    useState<SearchSourceProvider>("retrieval");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<string[]>([]);
  const [documents, setDocuments] = useState<SourceDocumentView[]>([]);
  const [messages, setMessages] = useState<ChatMessageView[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [streamingAnswer, setStreamingAnswer] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [adminSummary, setAdminSummary] = useState<AdminSurfaceSummary | null>(null);
  const [analyticsByLLM, setAnalyticsByLLM] = useState<BreakdownAnalytics | null>(null);
  const [analyticsByPersona, setAnalyticsByPersona] = useState<BreakdownAnalytics | null>(null);
  const [analyticsByFlow, setAnalyticsByFlow] = useState<BreakdownAnalytics | null>(null);
  const [showQueryHistory, setShowQueryHistory] = useState(false);
  const [showConnectors, setShowConnectors] = useState(false);
  const [showTools, setShowTools] = useState(false);
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
    setStreamingAnswer("");
    try {
      const activeSessionId = await ensureSession(controller.signal);
      const agentRequest: AgentExperienceRequest = {
        query: normalizedQuery,
        session_id: activeSessionId,
        search_url: searchUrl,
        top_k: topK,
        source_provider: sourceProvider,
      };

      const response: AgentExperienceResponse = await runAgent(agentRequest, {
        signal: controller.signal,
      });
      setSessionId(response.session_id);
      setAnswer(response.answer);
      setCitations(response.citations);
      setDocuments(response.documents);
      setMessages(response.messages);
      setIntent(response.intent);
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
  }, [ensureSession, query, searchUrl, sourceProvider, topK]);

  const handleNewSession = useCallback(async () => {
    requestRef.current?.abort();
    const session = await createSession({ title: "Search session" });
    setSessionId(session.id);
    setAnswer("");
    setCitations([]);
    setDocuments([]);
    setMessages([]);
    setError(null);
    setIntent(undefined);
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
            <button
              className={`icon-button${showConnectors ? " active" : ""}`}
              type="button"
              onClick={() => setShowConnectors((v) => !v)}
              title="Manage connectors"
            >
              <Plug size={18} />
              <span>Connectors</span>
            </button>
            <button
              className={`icon-button${showTools ? " active" : ""}`}
              type="button"
              onClick={() => setShowTools((v) => !v)}
              title="Manage tools"
            >
              <Wrench size={18} />
              <span>Tools</span>
            </button>
            <button
              className={`icon-button${showQueryHistory ? " active" : ""}`}
              type="button"
              onClick={() => setShowQueryHistory((v) => !v)}
              title="Query history audit"
            >
              <ClipboardList size={18} />
              <span>History</span>
            </button>
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
          sourceProvider={sourceProvider}
          isLoading={isLoading}
          onQueryChange={setQuery}
          onSearchUrlChange={setSearchUrl}
          onTopKChange={handleTopKChange}
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

        {showConnectors && <ConnectorPanel />}

        {showTools && <ToolPanel />}

        {showQueryHistory && <QueryHistoryPanel />}

        <div className={`results-layout${intent ? ` intent-${intent}` : ""}`}>
          <section className="answer-column" aria-label="Answer">
            <div className="section-heading">
              <Bot size={18} />
              <h2>Answer</h2>
            </div>
            <AnswerPanel
              answer={streamingAnswer || answer}
              citations={citations}
              intent={intent}
              documentCount={documents.length}
            />
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
