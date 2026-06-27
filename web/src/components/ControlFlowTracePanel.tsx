import { memo, useEffect, useMemo, useState } from "react";
import type { ControlFlowEventView } from "../types";

interface ControlFlowTracePanelProps {
  events: ControlFlowEventView[];
  live: boolean;
}

const COMPONENT_LABELS: Record<string, string> = {
  planner: "Planner",
  loop_controller: "Loop controller",
  search_tool: "Search tool",
  reranker_tool: "Reranker tool",
  evidence_judge: "Evidence judge",
  answer_generator: "Answer generator",
};

const ACTION_LABELS: Record<string, string> = {
  turn_parsed: "Turn parsed",
  search_planned: "Search planned",
  answer_planned: "Answer planned",
  format_recovery: "Format recovery",
  search_continued: "Continue searching",
  budget_exhausted: "Budget exhausted",
  plateau_stopped: "Evidence plateau",
  answer_accepted: "Answer accepted",
  answer_rejected: "Answer rejected",
  answer_forced: "Answer forced",
  vector_db_search: "Vector search",
  web_search: "Web search",
  fallback_search: "Fallback search",
  query_skipped: "Query skipped",
  rerank: "Reranked results",
  rerank_skipped: "Rerank skipped",
  evidence_evaluated: "Evidence evaluated",
  citations_resolved: "Citations resolved",
};

const STATUS_LABELS: Record<string, string> = {
  started: "In progress",
  completed: "Completed",
  decided: "Decided",
  skipped: "Skipped",
  failed: "Failed",
};

function detailsSummary(event: ControlFlowEventView): string {
  const parts: string[] = [];
  const { details } = event;
  if (typeof details.document_count === "number") {
    parts.push(`${details.document_count} docs`);
  }
  if (typeof details.citation_count === "number") {
    parts.push(`${details.citation_count} citations`);
  }
  if (typeof details.evidence_score === "number") {
    parts.push(`evidence ${details.evidence_score.toFixed(2)}`);
  }
  if (typeof details.query_count === "number") {
    parts.push(`${details.query_count} queries`);
  }
  if (typeof details.decision === "string") {
    parts.push(details.decision);
  }
  return parts.join(" · ");
}

export const ControlFlowTracePanel = memo(function ControlFlowTracePanel({
  events,
  live,
}: ControlFlowTracePanelProps) {
  const [expanded, setExpanded] = useState(live);
  const ordered = useMemo(
    () => [...events].sort((a, b) => a.sequence - b.sequence),
    [events],
  );

  useEffect(() => {
    setExpanded(live);
  }, [live]);

  if (ordered.length === 0) return null;

  if (!expanded) {
    return (
      <button
        type="button"
        className="control-flow-summary"
        aria-label={`show control flow — ${ordered.length} events`}
        onClick={() => setExpanded(true)}
      >
        <span>✓ {ordered.length} control-flow events</span>
        <span>show control flow ▸</span>
      </button>
    );
  }

  return (
    <section className="control-flow-trace" aria-label="Control flow">
      <div className="control-flow-header">
        <strong>Control flow</strong>
        {!live && (
          <button type="button" onClick={() => setExpanded(false)}>
            hide
          </button>
        )}
      </div>
      <ol aria-live={live ? "polite" : undefined}>
        {ordered.map((event) => {
          const summary = detailsSummary(event);
          return (
            <li key={event.sequence} className={`control-flow-event status-${event.status}`}>
              <span className="control-flow-sequence">{event.sequence}</span>
              <div>
                <div className="control-flow-event-title">
                  <strong>{COMPONENT_LABELS[event.component] ?? event.component}</strong>
                  <span>{ACTION_LABELS[event.action] ?? event.action}</span>
                </div>
                <div className="control-flow-event-meta">
                  <span>{STATUS_LABELS[event.status] ?? event.status}</span>
                  {event.duration_ms != null && <span>{event.duration_ms} ms</span>}
                  {summary && <span>{summary}</span>}
                </div>
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
});
