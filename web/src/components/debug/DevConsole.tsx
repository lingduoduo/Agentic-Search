import { RetrievalLab } from "./RetrievalLab";

/**
 * Dev-console container. Hosts observability panels for the backend servers.
 * Phase 1 ships the Retrieval Lab; health/workers/chat-trace panels follow.
 */
export function DevConsole() {
  return (
    <section className="dev-console" aria-label="Dev console">
      <RetrievalLab />
    </section>
  );
}
