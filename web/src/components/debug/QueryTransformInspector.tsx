import { useState } from "react";
import { runQueryTransform } from "../../api";
import type { QueryTransformResult } from "../../types";

const LEG_LABELS: Record<string, string> = {
  sub_queries: "decompose",
  multi_query: "multi-query",
  rewrite: "rewrite",
  hyde_text: "HyDE",
  step_back: "step-back",
  keywords: "keywords",
};

/**
 * Dev-console panel: run *only* the pre-retrieval query-transform pipeline and
 * show `original → variants` + merged filters + which legs fired. The per-mode
 * retrieval endpoints bypass the pipeline, so this is its own panel.
 */
export function QueryTransformInspector() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<QueryTransformResult | null>(null);
  const [running, setRunning] = useState(false);

  async function run() {
    if (!query.trim()) return;
    setRunning(true);
    setResult(null);
    try {
      setResult(await runQueryTransform(query));
    } catch {
      setResult(null);
    } finally {
      setRunning(false);
    }
  }

  const activeLegs = result
    ? Object.entries(result.legs).filter(([, v]) =>
        Array.isArray(v) ? v.length > 0 : Boolean(v),
      )
    : [];

  return (
    <section className="query-transform">
      <h2>Query Transform Inspector</h2>
      <div className="query-transform__controls">
        <label>
          Query
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="vector database"
          />
        </label>
        <button type="button" onClick={run} disabled={running}>
          {running ? "Running…" : "Transform"}
        </button>
      </div>

      {result && !result.active && (
        <p className="query-transform__inactive">
          no transform active — pipeline disabled or no LLM configured (variants =
          original)
        </p>
      )}

      {result && (
        <div className="query-transform__out">
          <p className="query-transform__original">
            original: <code>{result.original}</code>
          </p>
          <h3>Variants</h3>
          <ul className="query-transform__variants">
            {result.variants.map((v) => (
              <li key={v}>{v}</li>
            ))}
          </ul>
          {activeLegs.length > 0 && (
            <p className="query-transform__legs">
              active legs:{" "}
              {activeLegs.map(([k]) => LEG_LABELS[k] ?? k).join(", ")}
            </p>
          )}
          {Object.keys(result.merged_filters).length > 0 && (
            <p className="query-transform__filters">
              filters: <code>{JSON.stringify(result.merged_filters)}</code>
            </p>
          )}
        </div>
      )}
    </section>
  );
}
