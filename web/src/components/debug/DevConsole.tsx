import type { ControlFlowEventView } from "../../types";
import { EvalResultsPanel } from "./EvalResultsPanel";
import { QueryTransformInspector } from "./QueryTransformInspector";
import { RequestInspector } from "./RequestInspector";
import { RequestTracePanel } from "./RequestTracePanel";
import { RetrievalLab } from "./RetrievalLab";
import { ServerHealthGrid } from "./ServerHealthGrid";
import { WorkerMonitor } from "./WorkerMonitor";

interface Props {
  /** Last agent run — feeds the grounding debug ("sources but empty answer"). */
  answer: string;
  citations: string[];
  /** Last run's control-flow trace — feeds the Request Trace waterfall (F6 spine). */
  controlFlowTrace: ControlFlowEventView[];
  /** Last run's captured request id — auto-selects it in the Request Inspector. */
  selectedRequestId?: string | null;
}

/**
 * Dev-console container. Hosts observability panels for the backend servers.
 * Request Trace spine + Server Health/Grounding + Retrieval Lab; workers/chat follow.
 */
export function DevConsole({ answer, citations, controlFlowTrace, selectedRequestId }: Props) {
  return (
    <section className="dev-console" aria-label="Dev console">
      <RequestInspector selectedRequestId={selectedRequestId} />
      <RequestTracePanel events={controlFlowTrace} />
      <ServerHealthGrid answer={answer} citations={citations} />
      <WorkerMonitor />
      <QueryTransformInspector />
      <RetrievalLab />
      <EvalResultsPanel />
    </section>
  );
}
