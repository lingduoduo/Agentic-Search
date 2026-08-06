import { useState } from "react";
import type { CatalogServer, ToolDiscoverResult } from "../types";

interface Props {
  /** Tools grouped by server. `null` means still loading. */
  servers: CatalogServer[] | null;
  registeredCount: number;
  /** Last discovery result, or null if none has been run. */
  discovery: ToolDiscoverResult | null;
  onDiscover: (query: string) => void;
  /** Shown in place of the catalog — e.g. "requires admin". Not an error state. */
  note?: string;
}

/**
 * Read-only view of the tool registry: the catalog grouped by server, plus a
 * query box that ranks tools via the semantic router (TF-IDF, no LLM).
 *
 * Presentational only — each call site fetches its own data, because the two
 * sources differ: /admin/tools (always mounted, admin-gated, carries the
 * agent_callable / user_scoped flags) and /api/debug/tools (dev console only,
 * pre-grouped, no flags). One renderer keeps them from drifting apart the way
 * ToolPanel and debug/ToolsPanel did.
 */
export function ToolCatalog({
  servers,
  registeredCount,
  discovery,
  onDiscover,
  note,
}: Props) {
  const [query, setQuery] = useState("");

  function submit() {
    const trimmed = query.trim();
    if (trimmed) onDiscover(trimmed);
  }

  return (
    <section className="tools-panel" aria-label="Tool registry">
      <h2>Tool registry</h2>

      {note ? (
        <p className="tools-panel__note">{note}</p>
      ) : servers === null ? (
        <p className="tools-panel__count">Loading…</p>
      ) : (
        <>
          <p className="tools-panel__count">
            {registeredCount} registered across {servers.length}{" "}
            {servers.length === 1 ? "server" : "servers"}
          </p>
          {servers.length === 0 && (
            <p className="tools-panel__note">No tools are registered.</p>
          )}
          {servers.map((server) => (
            <article key={server.name} className="tools-panel__server">
              <header>{server.name}</header>
              <ul>
                {server.tools.map((t) => (
                  <li key={t.name}>
                    <strong>{t.name}</strong>{" "}
                    <span className="tools-panel__source">{t.source}</span>
                    {t.agent_callable === false && (
                      <span
                        className="tools-panel__flag"
                        title="Registered and directly invocable, but never offered to an agent loop — so the agent on this page cannot call it."
                      >
                        not offered to agents
                      </span>
                    )}
                    {t.user_scoped === true && (
                      <span
                        className="tools-panel__flag"
                        title="Backed by per-user storage. Withheld from anonymous callers so a write cannot land in a shared bucket."
                      >
                        needs sign-in
                      </span>
                    )}
                    <div>{t.description}</div>
                  </li>
                ))}
              </ul>
            </article>
          ))}
        </>
      )}

      <div className="tools-panel__discover">
        <input
          aria-label="Discovery query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
          }}
          placeholder="Test discovery for a query…"
        />
        <button type="button" onClick={submit}>
          Discover
        </button>
      </div>
      <p className="tools-panel__hint">
        Ranks tools by keyword similarity, with no model involved — which is not
        how the agent chooses. The agent is offered every agent-callable tool and
        picks one itself.
      </p>

      {discovery && (
        <ol className="tools-panel__results">
          {discovery.final_tools.length === 0 && <li>No tools matched.</li>}
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
