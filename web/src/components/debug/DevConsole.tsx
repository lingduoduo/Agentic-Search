import type { ControlFlowEventView } from "../../types";
import { QueryTransformInspector } from "./QueryTransformInspector";
import { RequestTracePanel } from "./RequestTracePanel";
import { RetrievalLab } from "./RetrievalLab";
import { ServerHealthGrid } from "./ServerHealthGrid";

interface Props {
  /** Last agent run — feeds the grounding debug ("sources but empty answer"). */
  answer: string;
  citations: string[];
  /** Last run's control-flow trace — feeds the Request Trace waterfall (F6 spine). */
  controlFlowTrace: ControlFlowEventView[];
}

/**
 * Dev-console container. Hosts observability panels for the backend servers.
 * Request Trace spine + Server Health/Grounding + Retrieval Lab; workers/chat follow.
 */
export function DevConsole({ answer, citations, controlFlowTrace }: Props) {
  return (
    <section className="dev-console" aria-label="Dev console">
      <RequestTracePanel events={controlFlowTrace} />
      <ServerHealthGrid answer={answer} citations={citations} />
      <QueryTransformInspector />
      <RetrievalLab />
    </section>
  );
}
