import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import { Bot, ClipboardList, FileSearch, Gauge, MessageSquarePlus, Plug, Search, Wrench } from "lucide-react";
import {
  createSession,
  getAdminSummary,
  getAnalyticsByFlow,
  getAnalyticsByLLM,
  getAnalyticsByPersona,
  streamAgent,
  submitToolApproval,
} from "./api";
import { AdminOverview } from "./components/AdminOverview";
import { ToolCallTracePanel } from "./components/ToolCallTracePanel";
import { AnalyticsDashboard } from "./components/AnalyticsDashboard";
import { AnswerPanel } from "./components/AnswerPanel";
import { ConnectorPanel } from "./components/ConnectorPanel";
import { DevConsole } from "./components/debug/DevConsole";
import { QueryHistoryPanel } from "./components/QueryHistoryPanel";
import { ToolPanel } from "./components/ToolPanel";
import { ToolApprovalCard } from "./components/ToolApprovalCard";
import { SearchComposer } from "./components/SearchComposer";
import { SessionTimeline } from "./components/SessionTimeline";
import { SourceGrid } from "./components/SourceGrid";
import { ToolAgentView } from "./components/ToolAgentView";
import type {
  AdminSurfaceSummary,
  AgentExperienceRequest,
  BreakdownAnalytics,
  ChatMessageView,
  ControlFlowEventView,
  ProgressStep,
  SearchSourceProvider,
  SourceDocumentView,
  ToolCallTraceView,
  ToolApprovalView,
} from "./types";

const DEFAULT_SEARCH_URL = "http://localhost:8001/retrieve";
const DEFAULT_BROWSER_SEARCH_URL = "http://localhost:8001/retrieve";

function upsertTrace(
  events: ControlFlowEventView[],
  event: ControlFlowEventView,
): ControlFlowEventView[] {
  return [...events.filter((item) => item.sequence !== event.sequence), event]
    .sort((a, b) => a.sequence - b.sequence);
}

// Dev escape hatch: `?dev=1` reveals the raw Retrieval URL override. In normal
// use the backend resolves the retrieval URL from the selected Source, so the
// client never dictates what the server fetches.
const DEV_MODE =
  typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).has("dev");

export function App() {
  const [query, setQuery] = useState("");
  const [searchUrl, setSearchUrl] = useState(DEFAULT_SEARCH_URL);
  const [topK, setTopK] = useState(5);
  const [intent, setIntent] = useState<"search" | "chat" | "tool" | undefined>(undefined);
  const [route, setRoute] = useState<string | undefined>(undefined);
  const [routeDegraded, setRouteDegraded] = useState<string | undefined>(undefined);
  const [sourceProvider, setSourceProvider] =
    useState<SearchSourceProvider>("auto");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<string[]>([]);
  const [documents, setDocuments] = useState<SourceDocumentView[]>([]);
  const [messages, setMessages] = useState<ChatMessageView[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [streamingAnswer, setStreamingAnswer] = useState("");
  const [progressSteps, setProgressSteps] = useState<ProgressStep[]>([]);
  const [completedSteps, setCompletedSteps] = useState<ProgressStep[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [adminSummary, setAdminSummary] = useState<AdminSurfaceSummary | null>(null);
  const [analyticsByLLM, setAnalyticsByLLM] = useState<BreakdownAnalytics | null>(null);
  const [analyticsByPersona, setAnalyticsByPersona] = useState<BreakdownAnalytics | null>(null);
  const [analyticsByFlow, setAnalyticsByFlow] = useState<BreakdownAnalytics | null>(null);
  const [toolCalls, setToolCalls] = useState<ToolCallTraceView[]>([]);
  const [controlFlowTrace, setControlFlowTrace] = useState<ControlFlowEventView[]>([]);
  const [lastRequestId, setLastRequestId] = useState<string | null>(null);
  const [pendingApprovals, setPendingApprovals] = useState<ToolApprovalView[]>([]);
  const [showQueryHistory, setShowQueryHistory] = useState(false);
  const [showConnectors, setShowConnectors] = useState(false);
  const [showTools, setShowTools] = useState(false);
  const [showConsole, setShowConsole] = useState(false);
  const [surface, setSurface] = useState<"assistant" | "tool">("assistant");
  // Dev-only observability console; gated at build time, never on in prod.
  const debugPanels = import.meta.env.VITE_DEBUG_PANELS === "1";
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
    // "Grounded" only when the answer is backed by retrieved sources; an
    // answer with no citations is parametric, so call it "Answered".
    if (answer) return citations.length > 0 ? "Grounded" : "Answered";
    return "Ready";
  }, [answer, citations, error, isLoading]);
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

  const handleSubmit = useCallback(async (eventOrQuery?: FormEvent | string) => {
    if (eventOrQuery && typeof eventOrQuery !== "string") eventOrQuery.preventDefault();
    const raw = typeof eventOrQuery === "string" ? eventOrQuery : query;
    const normalizedQuery = raw.trim();
    if (!normalizedQuery) return;

    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;

    setIsLoading(true);
    setError(null);
    setIntent(undefined);
    setRoute(undefined);
    setRouteDegraded(undefined);
    // Clear the previous turn's answer so a failed/next query never renders a
    // stale answer or citation anchors pointing at now-removed source cards.
    setAnswer("");
    setStreamingAnswer("");
    setCitations([]);
    setDocuments([]);
    setToolCalls([]);
    setProgressSteps([]);
    setCompletedSteps([]);
    setControlFlowTrace([]);
    setPendingApprovals([]);
    try {
      const activeSessionId = await ensureSession(controller.signal);
      // Append the user turn to the local timeline. Done after ensureSession so
      // the new-session reset inside it can't wipe the message we just added.
      setMessages((current) => [
        ...current,
        { role: "user", content: normalizedQuery },
      ]);
      const agentRequest: AgentExperienceRequest = {
        query: normalizedQuery,
        session_id: activeSessionId,
        // Only forwarded in dev mode; otherwise the backend resolves the URL.
        search_url: DEV_MODE ? searchUrl : undefined,
        top_k: topK,
        source_provider: DEV_MODE ? sourceProvider : undefined,
      };

      let accumulatedAnswer = "";
      let liveSteps: ProgressStep[] = [];
      for await (const event of streamAgent(agentRequest, { signal: controller.signal })) {
        if (event.type === "progress") {
          liveSteps = [...liveSteps, { turn: event.turn, text: event.text }];
          setProgressSteps(liveSteps);
        } else if (event.type === "trace") {
          setControlFlowTrace((current) => upsertTrace(current, event.event));
        } else if (event.type === "approval_required") {
          setPendingApprovals((current) => [
            ...current.filter((approval) => approval.id !== event.approval.id),
            event.approval,
          ]);
        } else if (event.type === "answer") {
          accumulatedAnswer = event.text;
          setStreamingAnswer(event.text);
        } else if (event.type === "done") {
          setSessionId(event.session_id);
          setCitations(event.citations);
          setDocuments(event.documents);
          setToolCalls(event.tool_calls ?? []);
          setControlFlowTrace(
            [...(event.control_flow_trace ?? [])].sort(
              (a, b) => a.sequence - b.sequence,
            ),
          );
          if (event.intent) setIntent(event.intent);
          setRoute(event.route ?? undefined);
          setRouteDegraded(event.route_degraded ?? undefined);
          setAnswer(accumulatedAnswer);
          setMessages((current) => [
            ...current,
            { role: "assistant", content: accumulatedAnswer },
          ]);
          setCompletedSteps(liveSteps);
          setProgressSteps([]);
          setPendingApprovals([]);
          setLastRequestId(event.request_id ?? null);
        } else if (event.type === "error") {
          throw new Error(event.detail);
        }
      }
    } catch (caught) {
      if (requestRef.current !== controller) return;
      setPendingApprovals([]);
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setError(caught instanceof Error ? caught.message : "Search failed");
      setAnswer("");
      setCitations([]);
      setDocuments([]);
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setIsLoading(false);
      }
    }
  }, [ensureSession, query, searchUrl, sourceProvider, topK]);

  const handleApprovalDecision = useCallback(async (
    approvalId: string,
    decision: "approve" | "deny",
  ) => {
    const signal = requestRef.current?.signal;
    await submitToolApproval(approvalId, decision, { signal });
    setPendingApprovals((current) =>
      current.filter((approval) => approval.id !== approvalId),
    );
  }, []);

  const handleNewSession = useCallback(async () => {
    requestRef.current?.abort();
    setPendingApprovals([]);
    const session = await createSession({ title: "Search session" });
    setSessionId(session.id);
    setAnswer("");
    setCitations([]);
    setDocuments([]);
    setToolCalls([]);
    setControlFlowTrace([]);
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
            {route && (
              <span
                className={`route-pill${routeDegraded ? " route-pill--degraded" : ""}`}
                title={
                  routeDegraded
                    ? `Routed to ${route} (degraded: ${routeDegraded})`
                    : `Routed to ${route}`
                }
              >
                via {route}
                {routeDegraded ? " ⚠" : ""}
              </span>
            )}
            <div className="surface-switcher" role="tablist" aria-label="Surface">
              <button
                role="tab"
                aria-selected={surface === "assistant"}
                className={`icon-button${surface === "assistant" ? " active" : ""}`}
                onClick={() => setSurface("assistant")}
              >
                Assistant
              </button>
              <button
                role="tab"
                aria-selected={surface === "tool"}
                className={`icon-button${surface === "tool" ? " active" : ""}`}
                onClick={() => setSurface("tool")}
              >
                Tool Agent
              </button>
            </div>
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
            {debugPanels && (
              <button
                className={`icon-button${showConsole ? " active" : ""}`}
                type="button"
                onClick={() => setShowConsole((v) => !v)}
                title="Dev console — backend observability"
              >
                <Gauge size={18} />
                <span>Console</span>
              </button>
            )}
            <button className="icon-button" type="button" onClick={handleNewSession}>
              <MessageSquarePlus size={18} />
              <span>New</span>
            </button>
          </div>
        </header>

        {surface === "assistant" && (
          <>
            <SearchComposer
              query={query}
              searchUrl={searchUrl}
              topK={topK}
              sourceProvider={sourceProvider}
              isLoading={isLoading}
              showUrlField={DEV_MODE}
              showSourcePicker={DEV_MODE}
              onQueryChange={setQuery}
              onSearchUrlChange={setSearchUrl}
              onTopKChange={handleTopKChange}
              onSourceProviderChange={handleSourceProviderChange}
              onSubmit={handleSubmit}
              onExampleSelect={(q) => {
                setQuery(q);
                handleSubmit(q);
              }}
            />

            {error && <div className="error-banner">{error}</div>}
          </>
        )}

        {debugPanels && showConsole && (
          <DevConsole
            answer={streamingAnswer || answer}
            citations={citations}
            controlFlowTrace={controlFlowTrace}
            selectedRequestId={lastRequestId}
          />
        )}

        {showConnectors && <ConnectorPanel />}

        {showTools && <ToolPanel />}

        {showQueryHistory && <QueryHistoryPanel />}

        {surface === "assistant" ? (
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
                progressSteps={progressSteps}
                completedSteps={completedSteps}
              />
              {pendingApprovals.map((approval) => (
                <ToolApprovalCard
                  key={approval.id}
                  approval={approval}
                  onDecision={(decision) =>
                    handleApprovalDecision(approval.id, decision)
                  }
                />
              ))}
              {intent === "tool" && <ToolCallTracePanel calls={toolCalls} />}
            </section>

            <section className="panel sources-panel wide" aria-label="Sources">
              <div className="section-heading">
                <Search size={18} />
                <h2>Sources</h2>
                <span className="count">{documents.length}</span>
              </div>
              <SourceGrid documents={documents} />
            </section>

            <section className="panel session-panel" aria-label="Session">
              <div className="section-heading">
                <MessageSquarePlus size={18} />
                <h2>Session</h2>
              </div>
              <SessionTimeline messages={messages} />
            </section>
          </div>
        ) : (
          <ToolAgentView />
        )}

        {adminSummary && <AdminOverview summary={adminSummary} />}
        {(analyticsByLLM || analyticsByPersona || analyticsByFlow) && (
          <AnalyticsDashboard
            byLLM={analyticsByLLM}
            byPersona={analyticsByPersona}
            byFlow={analyticsByFlow}
          />
        )}
      </section>
    </main>
  );
}
