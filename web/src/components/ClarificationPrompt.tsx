import type { ClarificationView } from "../types";

export function ClarificationPrompt({
  clarification,
  onSelect,
}: {
  clarification: ClarificationView;
  onSelect: (route: "chat" | "search" | "tool") => void;
}) {
  return (
    <section className="clarification" aria-label="Clarify the request">
      <p className="clarification-question">{clarification.question}</p>
      <div className="clarification-options">
        {clarification.options.map((option) => (
          <button
            key={option.route}
            type="button"
            className="icon-button"
            onClick={() => onSelect(option.route)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </section>
  );
}
