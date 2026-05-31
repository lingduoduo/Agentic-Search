import { useCallback, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import { Bot, FileSearch, MessageSquarePlus, Search } from "lucide-react";
import { createSession, runAgent } from "./api";
import { AdminOverview } from "./components/AdminOverview";
import { AnswerPanel } from "./components/AnswerPanel";
import { SearchComposer } from "./components/SearchComposer";
import { SessionTimeline } from "./components/SessionTimeline";
import { SourceGrid } from "./components/SourceGrid";
import type {
  AgentExperienceResponse,
  AdminSurfaceSummary,
  ChatMessageView,
  SourceDocumentView,
} from "./types";

const DEFAULT_SEARCH_URL = "http://localhost:8000/retrieve";

const ADMIN_SUMMARY: AdminSurfaceSummary = {
  healthLabel: "Operational readiness",
  healthScore: 94,
  metrics: [
    { label: "Connectors", value: "12", detail: "10 active syncs" },
    { label: "Indexing", value: "98.7%", detail: "freshness SLA" },
    { label: "Users/groups", value: "4.2k", detail: "SCIM synced" },
    { label: "Tools/actions", value: "18", detail: "7 MCP backed" },
  ],
  sections: [
    {
      key: "connectors",
      title: "Connector management",
      status: "Healthy",
      tone: "good",
      description: "Source syncs, credentials, crawl windows, and permissions sync.",
      items: ["Slack, Drive, GitHub, SharePoint", "Retry queue monitored"],
    },
    {
      key: "indexing",
      title: "Indexing status",
      status: "Watching",
      tone: "watch",
      description: "Fetch, parse, chunk, enrich, embed, and search-index pipeline.",
      items: ["Last batch 6 min ago", "2 delayed documents"],
    },
    {
      key: "access",
      title: "Users and groups",
      status: "Synced",
      tone: "good",
      description: "Internal groups, external group mappings, and document ACLs.",
      items: ["SCIM delta sync enabled", "ACL audits available"],
    },
    {
      key: "auth",
      title: "Auth controls",
      status: "Enforced",
      tone: "good",
      description: "SSO, API keys, tenant gates, role policies, and token limits.",
      items: ["Enterprise SSO active", "Admin changes tracked"],
    },
    {
      key: "models",
      title: "Model settings",
      status: "Ready",
      tone: "neutral",
      description: "Primary LLM, reasoning model, reranker, and embedding settings.",
      items: ["Fallback provider configured", "Budget guardrails enabled"],
    },
    {
      key: "tools",
      title: "Tools and actions",
      status: "Governed",
      tone: "good",
      description: "Custom actions, OpenAPI tools, MCP integrations, and policies.",
      items: ["Approval workflow required", "Tool traces retained"],
    },
    {
      key: "analytics",
      title: "Analytics",
      status: "Live",
      tone: "good",
      description: "Sessions, citations, answer quality, latency, and usage trends.",
      items: ["Citation coverage tracked", "P95 latency visible"],
    },
    {
      key: "enterprise",
      title: "Enterprise controls",
      status: "Active",
      tone: "good",
      description: "Licensing, tenant isolation, audit hooks, and data controls.",
      items: ["Tier gates enforced", "Audit exports enabled"],
    },
  ],
};

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
  const requestRef = useRef<AbortController | null>(null);

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
  }, [ensureSession, query, searchUrl, topK]);

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
          isLoading={isLoading}
          onQueryChange={setQuery}
          onSearchUrlChange={setSearchUrl}
          onTopKChange={handleTopKChange}
          onSubmit={handleSubmit}
        />

        {error && <div className="error-banner">{error}</div>}

        <AdminOverview summary={ADMIN_SUMMARY} />

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
