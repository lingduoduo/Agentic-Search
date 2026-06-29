import { useState } from "react";
import { runDebugRetrieval } from "../../api";
import type { DebugRetrievalOutcome, RetrievalMode } from "../../types";

const MODES: RetrievalMode[] = ["sparse", "dense", "hybrid", "graph"];

type Outcomes = Partial<Record<RetrievalMode, DebugRetrievalOutcome>>;

/**
 * Dev-console panel: run a query against each per-mode retrieval endpoint and
 * compare results side by side. Surfaces 404 (endpoint not mounted) and 503
 * (dense not configured) explicitly instead of as a generic failure.
 */
export function RetrievalLab() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [rrfK, setRrfK] = useState(60);
  const [mmrLambda, setMmrLambda] = useState(0.5);
  const [overFetch, setOverFetch] = useState(2);
  const [rerank, setRerank] = useState(false);
  const [rerankRequested, setRerankRequested] = useState(false);
  const [outcomes, setOutcomes] = useState<Outcomes>({});
  const [running, setRunning] = useState(false);

  const denseUnavailable = outcomes.dense?.status === 503;

  async function run() {
    if (!query.trim()) return;
    setRunning(true);
    setOutcomes({});
    setRerankRequested(rerank);
    const params = {
      query,
      top_k: topK,
      rrf_k: rrfK,
      mmr_lambda: mmrLambda,
      over_fetch: overFetch,
      rerank,
    };
    const results = await Promise.all(
      MODES.map((mode) => runDebugRetrieval(mode, params).then((o) => [mode, o] as const)),
    );
    setOutcomes(Object.fromEntries(results) as Outcomes);
    setRunning(false);
  }

  return (
    <section className="retrieval-lab">
      <h2>Retrieval Lab</h2>
      <div className="retrieval-lab__controls">
        <label>
          Query
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="vector database"
          />
        </label>
        <label>
          Top K
          <input
            type="number"
            value={topK}
            min={1}
            max={100}
            onChange={(e) => setTopK(Number(e.target.value))}
          />
        </label>
        <label>
          rrf_k
          <input
            type="number"
            value={rrfK}
            onChange={(e) => setRrfK(Number(e.target.value))}
          />
        </label>
        <label>
          mmr_lambda
          <input
            type="number"
            step={0.1}
            value={mmrLambda}
            onChange={(e) => setMmrLambda(Number(e.target.value))}
          />
        </label>
        <label>
          over_fetch
          <input
            type="number"
            value={overFetch}
            onChange={(e) => setOverFetch(Number(e.target.value))}
          />
        </label>
        <label className="retrieval-lab__rerank">
          <input
            type="checkbox"
            checked={rerank}
            onChange={(e) => setRerank(e.target.checked)}
          />
          rerank
        </label>
        <button type="button" onClick={run} disabled={running}>
          {running ? "Running…" : "Run"}
        </button>
      </div>

      {denseUnavailable && (
        <p className="retrieval-lab__warning">
          Dense leg unavailable — hybrid collapses to sparse-only fusion.
        </p>
      )}

      <div className="retrieval-lab__grid">
        {MODES.map((mode) => (
          <ModeColumn
            key={mode}
            mode={mode}
            outcome={outcomes[mode]}
            rerankRequested={rerankRequested}
          />
        ))}
      </div>
    </section>
  );
}

function ModeColumn({
  mode,
  outcome,
  rerankRequested,
}: {
  mode: RetrievalMode;
  outcome: DebugRetrievalOutcome | undefined;
  rerankRequested: boolean;
}) {
  const rerankInactive =
    rerankRequested &&
    outcome?.ok &&
    !outcome.data?.retrieval_mode.includes("+reranked");
  return (
    <div className="retrieval-lab__mode">
      <h3>{mode}</h3>
      {rerankInactive && (
        <p className="retrieval-lab__note">no reranker active</p>
      )}
      {!outcome && <p className="retrieval-lab__idle">—</p>}
      {outcome && outcome.status === 404 && (
        <p className="retrieval-lab__error">
          Endpoint not available (404). This server does not expose
          /internal/search/{mode}.
        </p>
      )}
      {outcome && outcome.status === 503 && (
        <p className="retrieval-lab__error">Dense not configured (503).</p>
      )}
      {outcome && !outcome.ok && outcome.status !== 404 && outcome.status !== 503 && (
        <p className="retrieval-lab__error">
          {outcome.detail ?? `Failed (${outcome.status})`}
        </p>
      )}
      {outcome?.ok && outcome.data && (
        <>
          <p className="retrieval-lab__meta">
            {outcome.data.retrieval_mode} · {outcome.data.latency_ms}ms
          </p>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>doc_id</th>
                <th>score</th>
                <th>title</th>
              </tr>
            </thead>
            <tbody>
              {outcome.data.results.map((row, i) => (
                <tr key={row.doc_id}>
                  <td>{i + 1}</td>
                  <td>{row.doc_id}</td>
                  <td>{row.score.toFixed(3)}</td>
                  <td>{row.title}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
