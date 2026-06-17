import { memo } from "react";
import type { ChatMessageView } from "../types";

interface SessionTimelineProps {
  messages: ChatMessageView[];
}

export const SessionTimeline = memo(function SessionTimeline({
  messages,
}: SessionTimelineProps) {
  const visible = messages
    .map((m, i) => ({ m, i }))
    .filter(({ m }) => m.role !== "system");

  if (visible.length === 0) {
    return <div className="empty-state compact">Start a query to create history.</div>;
  }

  return (
    <div className="chat-thread">
      {visible.map(({ m: message, i: origIdx }) => {
        const isUser = message.role === "user";
        const roundsUsed = message.metadata?.rounds_used as number | undefined;
        const numTurns = message.metadata?.num_turns as number | undefined;

        return (
          <div
            key={`${message.role}-${origIdx}`}
            className={`chat-row ${isUser ? "chat-row--user" : "chat-row--assistant"}`}
          >
            {isUser && <div className="chat-avatar chat-avatar--user">U</div>}

            <div className="chat-bubble-wrapper">
              <div className={`chat-bubble ${isUser ? "chat-bubble--user" : "chat-bubble--assistant"}`}>
                {message.content}
              </div>
              {!isUser && (roundsUsed != null || numTurns != null) && (
                <div className="chat-meta">
                  {roundsUsed != null && `${roundsUsed} rounds`}
                  {roundsUsed != null && numTurns != null && " · "}
                  {numTurns != null && `${numTurns} turns`}
                </div>
              )}
            </div>

            {!isUser && <div className="chat-avatar chat-avatar--assistant">A</div>}
          </div>
        );
      })}
    </div>
  );
});
