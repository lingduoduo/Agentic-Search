import { RetrievalLab } from "./RetrievalLab";
import { ServerHealthGrid } from "./ServerHealthGrid";

interface Props {
  /** Last agent run — feeds the grounding debug ("sources but empty answer"). */
  answer: string;
  citations: string[];
}

/**
 * Dev-console container. Hosts observability panels for the backend servers.
 * Ships the Retrieval Lab + Server Health / Grounding; workers/chat-trace follow.
 */
export function DevConsole({ answer, citations }: Props) {
  return (
    <section className="dev-console" aria-label="Dev console">
      <ServerHealthGrid answer={answer} citations={citations} />
      <RetrievalLab />
    </section>
  );
}
