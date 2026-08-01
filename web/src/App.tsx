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
import { ToolPanel } from "./components/ToolPanel";
import { AssistPage } from "./pages/AssistPage";
import type { AdminSurfaceSummary, BreakdownAnalytics } from "./types";

export function App() {
  const [adminSummary, setAdminSummary] = useState<AdminSurfaceSummary | null>(null);
  const [analyticsByLLM, setAnalyticsByLLM] = useState<BreakdownAnalytics | null>(null);
  const [analyticsByPersona, setAnalyticsByPersona] = useState<BreakdownAnalytics | null>(null);
  const [analyticsByFlow, setAnalyticsByFlow] = useState<BreakdownAnalytics | null>(null);
  const [showQueryHistory, setShowQueryHistory] = useState(false);
  const [showTools, setShowTools] = useState(false);
  const [surface, setSurface] = useState<"assistant" | "search" | "chat" | "tool">("assistant");

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
                aria-selected={surface === "search"}
                className={`icon-button${surface === "search" ? " active" : ""}`}
                onClick={() => setSurface("search")}
              >
                Search
              </button>
              <button
                role="tab"
                aria-selected={surface === "chat"}
                className={`icon-button${surface === "chat" ? " active" : ""}`}
                onClick={() => setSurface("chat")}
              >
                Chat
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
          </div>
        </header>

        {showTools && <ToolPanel />}

        {showQueryHistory && <QueryHistoryPanel />}

        {surface === "assistant" ? (
          <AssistPage />
        ) : surface === "search" ? (
          <SearchView />
        ) : surface === "chat" ? (
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
