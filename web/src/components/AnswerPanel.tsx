import { memo, useMemo, useState } from "react";
import type { ProgressStep } from "../types";

interface AnswerPanelProps {
  answer: string;
  citations: string[];
  intent?: "search" | "chat" | "tool";
  documentCount?: number;
  toolCallCount?: number;
  progressSteps?: ProgressStep[];
  completedSteps?: ProgressStep[];
}

function IntentBadge({ intent, citations, documentCount, toolCallCount }: {
  intent: "search" | "chat" | "tool";
  citations: string[];
  documentCount?: number;
  toolCallCount?: number;
}) {
  if (intent === "search") {
    const n = documentCount ?? 0;
    return <span className="intent-badge">Searched · {n} {n === 1 ? "source" : "sources"}</span>;
  }
  if (intent === "chat") {
    const n = citations.length;
    return <span className="intent-badge">Answered · {n} {n === 1 ? "citation" : "citations"}</span>;
  }
  return <span className="intent-badge">Used tools</span>;
}

function ProgressLog({
  steps,
  completedSteps,
}: {
  steps: ProgressStep[];
  completedSteps: ProgressStep[];
}) {
  const [expanded, setExpanded] = useState(false);

  if (steps.length > 0) {
    return (
      <div className="progress-log">
        <div className="progress-log-header">Agent reasoning</div>
        {steps.map((s, i) => (
          <div
            key={`${s.turn}-${i}`}
            className={`progress-step ${s.text.includes("writing") ? "active" : "done"}`}
          >
            {s.text.includes("writing") ? "⟳" : "✓"} Turn {s.turn} · {s.text}
          </div>
        ))}
      </div>
    );
  }

  if (completedSteps.length > 0) {
    const n = completedSteps.length;
    if (!expanded) {
      return (
        <button
          className="progress-summary"
          type="button"
          onClick={() => setExpanded(true)}
          aria-label={`show reasoning — ${n} ${n === 1 ? "turn" : "turns"}`}
        >
          <span>&#10003; {n} {n === 1 ? "turn" : "turns"}</span>
          <span className="show-reasoning">show reasoning &#9658;</span>
        </button>
      );
    }
    return (
      <div className="progress-log">
        <div className="progress-log-header">
          Agent reasoning
          <button
            type="button"
            onClick={() => setExpanded(false)}
            className="collapse-btn"
            aria-label="hide reasoning"
          >
            &#9660; hide
          </button>
        </div>
        {completedSteps.map((s, i) => (
          <div key={`${s.turn}-${i}`} className="progress-step done">
            &#10003; Turn {s.turn} · {s.text}
          </div>
        ))}
      </div>
    );
  }

  return null;
}

export const AnswerPanel = memo(function AnswerPanel({
  answer,
  citations,
  intent,
  documentCount,
  toolCallCount,
  progressSteps = [],
  completedSteps = [],
}: AnswerPanelProps) {
  const paragraphs = useMemo(() => answer.split(/\n\n+/).filter(Boolean), [answer]);

  if (!answer && progressSteps.length === 0) {
    return (
      <div className="empty-state">
        Results will appear here once the agent retrieves context.
      </div>
    );
  }

  return (
    <article className="answer-panel">
      <ProgressLog steps={progressSteps} completedSteps={completedSteps} />
      {intent && answer && (
        <IntentBadge
          intent={intent}
          citations={citations}
          documentCount={documentCount}
          toolCallCount={toolCallCount}
        />
      )}
      {paragraphs.map((para, i) => (
        <p key={i}>{para}</p>
      ))}
      {citations.length > 0 && (
        <div className="citation-row" aria-label="Citations">
          {citations.map((citation) => (
            <span key={citation}>{citation}</span>
          ))}
        </div>
      )}
    </article>
  );
});
