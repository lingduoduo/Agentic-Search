import { useState } from "react";
import type { ControlFlowEventView } from "../../types";

interface Props {
  events: ControlFlowEventView[];
}

/**
 * Request Trace (F6 spine) — a timed waterfall over the existing
 * `control_flow_trace`: one bar per event, width ∝ `duration_ms`, colored by
 * status, grouped by turn. Click a bar to drill into its `details`.
 */
export function RequestTracePanel({ events }: Props) {
  const [selected, setSelected] = useState<number | null>(null);

  if (events.length === 0) {
    return (
      <section className="request-trace">
        <h2>Request Trace</h2>
        <p className="request-trace__empty">no trace yet</p>
      </section>
    );
  }

  const maxDuration = Math.max(1, ...events.map((e) => e.duration_ms ?? 0));
  const open = events.find((e) => e.sequence === selected);

  return (
    <section className="request-trace">
      <h2>Request Trace</h2>
      <div className="request-trace__rows">
        {events.map((e) => {
          const pct =
            e.duration_ms && maxDuration
              ? Math.max(2, (e.duration_ms / maxDuration) * 100)
              : 2;
          return (
            <button
              key={e.sequence}
              type="button"
              className={`request-trace__row request-trace__row--${e.status}${
                selected === e.sequence ? " request-trace__row--selected" : ""
              }`}
              onClick={() =>
                setSelected(selected === e.sequence ? null : e.sequence)
              }
            >
              <span className="request-trace__label">
                T{e.turn} · {e.component} · {e.action}
              </span>
              <span className="request-trace__track">
                <span
                  data-testid={`trace-bar-${e.sequence}`}
                  className="request-trace__bar"
                  style={{ width: `${pct}%` }}
                />
              </span>
              <span className="request-trace__dur">
                {e.duration_ms != null ? `${e.duration_ms}ms` : "—"}
              </span>
            </button>
          );
        })}
      </div>

      {open && (
        <div className="request-trace__details">
          <h3>
            {open.component} · {open.action} ({open.status})
          </h3>
          {Object.keys(open.details).length === 0 ? (
            <p className="request-trace__no-details">no details</p>
          ) : (
            <table>
              <tbody>
                {Object.entries(open.details).map(([k, v]) => (
                  <tr key={k}>
                    <td className="request-trace__detail-key">{k}</td>
                    <td>{String(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </section>
  );
}
