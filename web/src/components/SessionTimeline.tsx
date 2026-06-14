import { memo } from "react";
import type { ChatMessageView } from "../types";

interface SessionTimelineProps {
  messages: ChatMessageView[];
}

export const SessionTimeline = memo(function SessionTimeline({
  messages,
}: SessionTimelineProps) {
  if (messages.length === 0) {
    return <div className="empty-state compact">Start a query to create history.</div>;
  }

  return (
    <ol className="timeline">
      {messages.map((message, index) => {
        const roundsUsed = message.metadata?.rounds_used as number | undefined;
        const numTurns = message.metadata?.num_turns as number | undefined;
        return (
          <li key={`${message.role}-${index}`}>
            <strong>{message.role}</strong>
            {message.role === "assistant" && (roundsUsed != null || numTurns != null) && (
              <span className="message-badge">
                {roundsUsed != null && `${roundsUsed} rounds`}
                {roundsUsed != null && numTurns != null && " · "}
                {numTurns != null && `${numTurns} turns`}
              </span>
            )}
            <p>{message.content}</p>
          </li>
        );
      })}
    </ol>
  );
});
