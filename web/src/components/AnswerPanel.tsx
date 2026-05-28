interface AnswerPanelProps {
  answer: string;
  citations: string[];
}

export function AnswerPanel({ answer, citations }: AnswerPanelProps) {
  if (!answer) {
    return (
      <div className="empty-state">
        Results will appear here once the agent retrieves context.
      </div>
    );
  }

  const paragraphs = answer.split(/\n\n+/).filter(Boolean);

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
}
