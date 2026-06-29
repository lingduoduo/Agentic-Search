import { useEffect, useState } from "react";
import { getWorkerMetrics } from "../../api";
import type { WorkerMetrics } from "../../types";

const CARDS: { key: keyof WorkerMetrics; testid: string; label: string }[] = [
  { key: "pending_index_attempts", testid: "pending", label: "Pending index" },
  {
    key: "in_progress_index_attempts",
    testid: "in_progress",
    label: "In progress",
  },
  { key: "total_documents", testid: "documents", label: "Documents" },
  { key: "active_connectors", testid: "connectors", label: "Connectors" },
];

/**
 * Dev-console panel: live indexing-pipeline snapshot (queue depth, docs,
 * connectors) computed on demand from the store. Complements the existing
 * ConnectorPanel (source side) with the pipeline side; "no data yet" when the
 * store isn't wired.
 */
export function WorkerMonitor() {
  const [metrics, setMetrics] = useState<WorkerMetrics | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let alive = true;
    getWorkerMetrics().then(
      (r) => {
        if (alive) {
          setMetrics(r.metrics);
          setLoaded(true);
        }
      },
      () => {
        if (alive) setLoaded(true);
      },
    );
    return () => {
      alive = false;
    };
  }, []);

  return (
    <section className="worker-monitor">
      <h2>Indexing / Workers</h2>
      {loaded && metrics === null && (
        <p className="worker-monitor__empty">no data yet</p>
      )}
      {metrics && (
        <>
          <div className="worker-monitor__grid">
            {CARDS.map((c) => (
              <div
                key={c.testid}
                data-testid={`metric-${c.testid}`}
                className="worker-monitor__card"
              >
                <span className="worker-monitor__value">{metrics[c.key]}</span>
                <span className="worker-monitor__label">{c.label}</span>
              </div>
            ))}
          </div>
          <p className="worker-monitor__meta">
            {metrics.process_memory_mb.toFixed(0)} MB · snapshot{" "}
            {metrics.timestamp}
          </p>
        </>
      )}
    </section>
  );
}
