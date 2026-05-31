import { memo, useMemo } from "react";

interface AnswerPanelProps {
  answer: string;
  citations: string[];
}

export const AnswerPanel = memo(function AnswerPanel({
  answer,
  citations,
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
