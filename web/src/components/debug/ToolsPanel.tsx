import { useEffect, useState } from "react";
import { discoverTools, getDebugTools } from "../../api";
import type { CatalogServer, ToolDiscoverResult } from "../../types";

/**
 * Dev-console panel: read-only view of the tool registry — the discovery
 * catalog grouped by server, plus a query box to test semantic tool discovery.
 */
export function ToolsPanel() {
  const [catalog, setCatalog] = useState<CatalogServer[] | null>(null);
  const [registeredCount, setRegisteredCount] = useState(0);
  const [query, setQuery] = useState("");
  const [discovery, setDiscovery] = useState<ToolDiscoverResult | null>(null);

  useEffect(() => {
    let alive = true;
    getDebugTools().then(
      (r) => {
        if (!alive) return;
        setCatalog(r.catalog);
        setRegisteredCount(r.registered.length);
      },
      () => alive && setCatalog([]),
    );
    return () => {
      alive = false;
    };
  }, []);

  function runDiscover() {
    if (!query.trim()) return;
    discoverTools(query).then(setDiscovery, () => setDiscovery(null));
  }

  return (
    <section className="tools-panel" aria-label="Tool registry">
      <h2>Tools</h2>
      <p className="tools-panel__count">{registeredCount} registered</p>
      {catalog?.map((server) => (
        <article key={server.name} className="tools-panel__server">
          <header>{server.name}</header>
          <ul>
            {server.tools.map((t) => (
              <li key={t.name}>
                <strong>{t.name}</strong>{" "}
                <span className="tools-panel__source">{t.source}</span>
                <div>{t.description}</div>
              </li>
            ))}
          </ul>
        </article>
      ))}
      <div className="tools-panel__discover">
        <input
          aria-label="Discovery query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Test discovery for a query…"
        />
        <button onClick={runDiscover}>Discover</button>
      </div>
      {discovery && (
        <ol className="tools-panel__results">
          {discovery.final_tools.map((t) => (
            <li key={t.name}>
              {t.name}{" "}
              <span>
                ({t.server}, {t.score.toFixed(3)})
              </span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
