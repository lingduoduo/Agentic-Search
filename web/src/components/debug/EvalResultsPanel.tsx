import { useEffect, useState } from "react";
import { getEvalResults } from "../../api";
import type { EvalResultFile } from "../../types";

function fmt(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(4);
}

/**
 * Dev-console panel: read-only view of offline evaluation results (BEIR / RAGAS
 * / retrieval / Bamboogle summaries) from the configured results directory.
 */
export function EvalResultsPanel() {
  const [results, setResults] = useState<EvalResultFile[] | null>(null);

  useEffect(() => {
    let alive = true;
    getEvalResults().then(
      (r) => alive && setResults(r.results),
      () => alive && setResults([]),
    );
    return () => {
      alive = false;
    };
  }, []);

  return (
    <section className="eval-results" aria-label="Evaluation results">
      <h2>Evaluation Results</h2>
      {results !== null && results.length === 0 && (
        <p className="eval-results__empty">
          No eval results yet — run an eval with <code>--output</code> into{" "}
          <code>data/eval/</code>.
        </p>
      )}
      {results?.map((file) => (
        <article key={file.name} className="eval-results__card">
          <header>
            <span className="eval-results__name">{file.name}</span>
            <span className="eval-results__mtime">
              {new Date(file.modified * 1000).toISOString().slice(0, 19).replace("T", " ")}
            </span>
          </header>
          {Object.keys(file.metrics).length === 0 ? (
            <p className="eval-results__empty">no numeric metrics</p>
          ) : (
            <table>
              <tbody>
                {Object.entries(file.metrics).map(([k, v]) => (
                  <tr key={k}>
                    <td>{k}</td>
                    <td>{fmt(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </article>
      ))}
    </section>
  );
}
