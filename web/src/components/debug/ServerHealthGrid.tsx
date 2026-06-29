import { useEffect, useState } from "react";
import { getServerHealth } from "../../api";
import type { ServerHealth } from "../../types";

interface Props {
  /** Last agent run's answer text + citations — for the grounding debug. */
  answer: string;
  citations: string[];
}

/**
 * Dev-console panel: server reachability grid + grounding debug.
 * Grounding directly explains the "sources but empty answer" case by separating
 * "did retrieval ground (citations)?" from "did the answer leg produce text?".
 */
export function ServerHealthGrid({ answer, citations }: Props) {
  const [servers, setServers] = useState<ServerHealth[]>([]);
  const [error, setError] = useState(false);

  useEffect(() => {
    let alive = true;
    getServerHealth().then(
      (r) => {
        if (alive) setServers(r.servers);
      },
      () => {
        if (alive) setError(true);
      },
    );
    return () => {
      alive = false;
    };
  }, []);

  const grounded = citations.length > 0;
  const answered = answer.trim().length > 0;
  const groundingLabel = answered
    ? grounded
      ? "grounded answer"
      : "answer, ungrounded"
    : grounded
      ? "grounded, no answer"
      : "no run yet";

  return (
    <section className="server-health">
      <h2>Server health</h2>
      {error && <p className="server-health__error">health check failed</p>}
      <div className="server-health__grid">
        {servers.map((s) => (
          <div
            key={s.name}
            data-testid={`health-${s.name}`}
            className={`server-health__cell server-health__cell--${s.status}`}
          >
            <span className="server-health__name">{s.name}</span>
            <span className="server-health__status">{s.status}</span>
            <span className="server-health__url">{s.url}</span>
          </div>
        ))}
      </div>

      <h3>Grounding (last run)</h3>
      <p
        className={`server-health__grounding server-health__grounding--${
          answered && grounded
            ? "ok"
            : answered || grounded
              ? "warn"
              : "idle"
        }`}
      >
        {groundingLabel}
        <span className="server-health__grounding-detail">
          {" "}
          — citations: {citations.length}, answer: {answered ? "yes" : "no"}
        </span>
      </p>
    </section>
  );
}
