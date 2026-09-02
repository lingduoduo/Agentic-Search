import { useEffect, useState } from "react";
import { getRouteLatency } from "../../api";
import type { RouteLatencyRow } from "../../types";

/**
 * Milliseconds, or a dash.
 *
 * A finite-guard here rather than a bare `toFixed`: #437 shipped a crash from a
 * non-finite metric crossing FastAPI -> JS as null and reaching toFixed.
 */
function ms(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(1) : "—";
}

/**
 * Dev-console panel: per-route request latency, slowest p95 first.
 *
 * The Request Inspector shows where one request spent its time. This shows
 * which route to inspect.
 */
export function LatencyPanel() {
  const [routes, setRoutes] = useState<RouteLatencyRow[] | null>(null);

  useEffect(() => {
    let alive = true;
    getRouteLatency().then(
      (r) => alive && setRoutes(r.routes),
      () => alive && setRoutes([]),
    );
    return () => {
      alive = false;
    };
  }, []);

  return (
    <section className="latency-panel" aria-label="Route latency">
      <h2>Route Latency</h2>
      {routes !== null && routes.length === 0 && (
        <p className="latency-panel__empty">
          No requests recorded yet — issue a request, then reload this panel.
        </p>
      )}
      {routes !== null && routes.length > 0 && (
        <table className="latency-panel__table">
          <thead>
            <tr>
              <th scope="col">Route</th>
              <th scope="col">Calls</th>
              <th scope="col">Errors</th>
              <th scope="col">p50 ms</th>
              <th scope="col">p95 ms</th>
              <th scope="col">max ms</th>
            </tr>
          </thead>
          <tbody>
            {routes.map((row) => (
              <tr key={`${row.method} ${row.route}`}>
                <th scope="row">
                  <code>
                    {row.method} {row.route}
                  </code>
                </th>
                <td>{row.count}</td>
                <td className={row.errors > 0 ? "latency-panel__errors" : undefined}>
                  {row.errors}
                </td>
                <td>{ms(row.p50_ms)}</td>
                <td>{ms(row.p95_ms)}</td>
                <td>{ms(row.max_ms)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
