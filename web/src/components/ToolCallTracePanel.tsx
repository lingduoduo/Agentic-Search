import { memo } from "react";
import { Wrench } from "lucide-react";
import type { ToolCallTraceView } from "../types";

interface ToolCallTracePanelProps {
  calls: ToolCallTraceView[];
}

export const ToolCallTracePanel = memo(function ToolCallTracePanel({
  calls,
}: ToolCallTracePanelProps) {
  if (calls.length === 0) return null;

  return (
    <section className="panel tool-trace-panel" aria-label="Tool calls">
      <div className="section-heading">
        <Wrench size={18} />
        <h2>Tool Calls</h2>
        <span className="count">{calls.length}</span>
      </div>
      <div className="tool-trace-list">
        {calls.map((call, i) => (
          <div
            key={i}
            className={`tool-trace-card ${call.status === "failed" ? "tool-trace-card--failed" : ""}`}
          >
            <div className="tool-trace-header">
              <span className={`tool-trace-status ${call.status === "failed" ? "tool-trace-status--failed" : "tool-trace-status--ok"}`}>
                {call.status === "completed" ? "✓" : "✗"}
              </span>
              <strong className="tool-trace-name">{call.tool_name}</strong>
              <span className="tool-trace-latency">{call.latency_ms} ms</span>
            </div>

            <div className="tool-trace-section">
              <span className="tool-trace-label">Arguments</span>
              <code className="tool-trace-code">
                {JSON.stringify(call.arguments, null, 2)}
              </code>
            </div>

            {call.status === "completed" ? (
              <div className="tool-trace-section">
                <span className="tool-trace-label">Result</span>
                <code className="tool-trace-code">{call.result_summary || "—"}</code>
              </div>
            ) : (
              <div className="tool-trace-section">
                <span className="tool-trace-label">Error</span>
                <code className="tool-trace-code tool-trace-code--error">
                  {call.error || "Unknown error"}
                </code>
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
});
