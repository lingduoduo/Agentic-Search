import { memo, useMemo } from "react";

interface AnswerPanelProps {
  answer: string;
  citations: string[];
  intent?: "search" | "chat" | "tool";
  documentCount?: number;
  toolCallCount?: number;
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

export const AnswerPanel = memo(function AnswerPanel({
  answer,
  citations,
  intent,
  documentCount,
  toolCallCount,
}: AnswerPanelProps) {
  const paragraphs = useMemo(() => answer.split(/\n\n+/).filter(Boolean), [answer]);

  if (!answer) {
    return (
      <div className="empty-state">
        Results will appear here once the agent retrieves context.
      </div>
    );
  }

  return (
    <article className="answer-panel">
      {intent && (
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
