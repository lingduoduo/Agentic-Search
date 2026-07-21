import { useState } from "react";
import { sendSearchMessage } from "../api";
import type { SearchDocView, SourceDocumentView } from "../types";
import { SourceGrid } from "./SourceGrid";

export function SearchView() {
  const [query, setQuery] = useState("");
  const [docs, setDocs] = useState<SearchDocView[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    const q = query.trim();
    if (!q || busy) return;
    setBusy(true); setError(null); setDocs([]);
    try {
      const r = await sendSearchMessage({ search_query: q });
      if (r.error) setError(r.error);
      setDocs(r.search_docs ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setBusy(false);
    }
  }

  const documents: SourceDocumentView[] = docs.map((d, i) => ({
    id: `D${i + 1}`,
    citation: d.title ?? `D${i + 1}`,
    title: d.title ?? "",
    content: d.content,
    url: d.url,
    score: d.score,
    metadata: d.metadata,
  }));

  return (
    <section className="search-view" aria-label="Search">
      <div className="search-view__composer">
        <input
          aria-label="Search query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder="Search the corpus…"
          disabled={busy}
        />
        <button onClick={submit} disabled={busy}>{busy ? "…" : "Search"}</button>
      </div>
      {error && <div className="error-banner">{error}</div>}
      <SourceGrid documents={documents} />
    </section>
  );
}
