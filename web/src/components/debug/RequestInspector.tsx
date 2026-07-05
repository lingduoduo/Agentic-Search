import { useCallback, useEffect, useState } from "react";
import { getDebugRequest, listDebugRequests } from "../../api";
import type { RequestSnapshot, RequestSummary } from "../../types";

const STAGE_ORDER = ["intent", "search", "tool", "llm", "final"] as const;

interface Props {
  /** Auto-select the run that just finished streaming. */
  selectedRequestId?: string | null;
}

export function RequestInspector({ selectedRequestId }: Props) {
  const [runs, setRuns] = useState<RequestSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [snap, setSnap] = useState<RequestSnapshot | null>(null);

  const refresh = useCallback(async () => {
    try {
      const { requests } = await listDebugRequests();
      setRuns(requests);
    } catch {
      setRuns([]);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh, selectedRequestId]);

  useEffect(() => {
    const id = selectedRequestId ?? selected;
    if (!id) return;
    let cancelled = false;
    getDebugRequest(id)
      .then((s) => !cancelled && setSnap(s))
      .catch(() => !cancelled && setSnap(null));
    return () => {
      cancelled = true;
    };
  }, [selected, selectedRequestId]);

  const orderedStages = snap
    ? [...snap.stages].sort(
        (a, b) => STAGE_ORDER.indexOf(a.stage) - STAGE_ORDER.indexOf(b.stage),
      )
    : [];

  return (
    <div className="request-inspector" aria-label="Request inspector">
      <div className="request-inspector__list">
        <button type="button" onClick={() => void refresh()}>
          Refresh
        </button>
        <ul>
          {runs.map((r) => (
            <li key={r.request_id}>
              <button type="button" onClick={() => setSelected(r.request_id)}>
                <span>{r.query || "(empty)"}</span>
                <span>{r.route ?? "?"}</span>
                <span>{r.stage_count} stages</span>
              </button>
            </li>
          ))}
          {runs.length === 0 && <li>No captured runs (enable debug panels).</li>}
        </ul>
      </div>
      <div className="request-inspector__detail">
        {snap ? (
          orderedStages.map((s, i) => (
            <details key={`${s.stage}-${i}`} open>
              <summary>
                {s.stage} · {s.label}
                {s.duration_ms != null ? ` · ${s.duration_ms.toFixed(0)}ms` : ""}
              </summary>
              <pre>{JSON.stringify(s.payload, null, 2)}</pre>
            </details>
          ))
        ) : (
          <p>Select a run to inspect its stages.</p>
        )}
      </div>
    </div>
  );
}
