import { useEffect, useState } from "react";
import { ClipboardList, FileSearch, Wrench } from "lucide-react";
import {
  getAdminSummary,
  getAnalyticsByFlow,
  getAnalyticsByLLM,
  getAnalyticsByPersona,
} from "./api";
import { AdminOverview } from "./components/AdminOverview";
import { AnalyticsDashboard } from "./components/AnalyticsDashboard";
import { ChatView } from "./components/ChatView";
import { QueryHistoryPanel } from "./components/QueryHistoryPanel";
import { SearchView } from "./components/SearchView";
import { ToolAgentView } from "./components/ToolAgentView";
import { ToolAdminPanel } from "./components/ToolAdminPanel";
import { AssistPage } from "./pages/AssistPage";
import { NavLink, useCanonicalRoute } from "./router";
import type { AdminSurfaceSummary, BreakdownAnalytics } from "./types";

export function App() {
  const [adminSummary, setAdminSummary] = useState<AdminSurfaceSummary | null>(null);
  const [analyticsByLLM, setAnalyticsByLLM] = useState<BreakdownAnalytics | null>(null);
  const [analyticsByPersona, setAnalyticsByPersona] = useState<BreakdownAnalytics | null>(null);
  const [analyticsByFlow, setAnalyticsByFlow] = useState<BreakdownAnalytics | null>(null);
  const [showQueryHistory, setShowQueryHistory] = useState(false);
  const [showTools, setShowTools] = useState(false);
  const route = useCanonicalRoute();

  useEffect(() => {
    getAdminSummary().then(setAdminSummary).catch(() => undefined);
    getAnalyticsByLLM().then(setAnalyticsByLLM).catch(() => undefined);
    getAnalyticsByPersona().then(setAnalyticsByPersona).catch(() => undefined);
    getAnalyticsByFlow().then(setAnalyticsByFlow).catch(() => undefined);
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
            <nav className="surface-nav" aria-label="Surfaces">
              <NavLink to="/assist" className="icon-button">Assistant</NavLink>
              <NavLink to="/search" className="icon-button">Search</NavLink>
              <NavLink to="/chat" className="icon-button">Chat</NavLink>
              <NavLink to="/tools" className="icon-button">Tools</NavLink>
            </nav>
            <button
              className={`icon-button${showTools ? " active" : ""}`}
              type="button"
              onClick={() => setShowTools((v) => !v)}
              title="Manage tools"
            >
              <Wrench size={18} />
              {/* Not "Tools": the nav link above already owns that label and
                  goes somewhere else entirely (the agent surface). */}
              <span>Manage tools</span>
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
          </div>
        </header>

        {showTools && <ToolAdminPanel />}

        {showQueryHistory && <QueryHistoryPanel />}

        {route === "/assist" ? (
          <AssistPage />
        ) : route === "/search" ? (
          <SearchView />
        ) : route === "/chat" ? (
          <ChatView />
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
