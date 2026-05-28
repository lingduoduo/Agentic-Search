import type { ChatMessageView } from "../types";

interface SessionTimelineProps {
  messages: ChatMessageView[];
}

export function SessionTimeline({ messages }: SessionTimelineProps) {
  if (messages.length === 0) {
    return <div className="empty-state compact">Start a query to create history.</div>;
  }

  return (
    <ol className="timeline">
      {messages.map((message, index) => (
        <li key={`${message.role}-${index}`}>
          <strong>{message.role}</strong>
          <p>{message.content}</p>
        </li>
      ))}
    </ol>
  );
}
