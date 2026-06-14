import { memo } from "react";
import { ExternalLink } from "lucide-react";
import type { SourceDocumentView } from "../types";

interface SourceGridProps {
  documents: SourceDocumentView[];
}

function scoreColor(score: number): string {
  if (score >= 0.7) return "rgb(34, 197, 94)";
  if (score >= 0.4) return "rgb(234, 179, 8)";
  if (score > 0)   return "rgb(249, 115, 22)";
  return "rgb(148, 163, 184)";
}

const SOURCE_COLORS: Record<string, string> = {
  "Browser Retrieval": "rgb(59, 130, 246)",
  "SerpAPI":           "rgb(139, 92, 246)",
  "Local Retrieval":   "rgb(107, 114, 128)",
  "All Active Sources":"rgb(14, 165, 233)",
};

export const SourceGrid = memo(function SourceGrid({ documents }: SourceGridProps) {
  if (documents.length === 0) {
    return <div className="empty-state compact">No sources yet.</div>;
  }

  return (
    <div className="source-grid">
      {documents.map((document, idx) => {
        const source =
          typeof document.metadata.source === "string"
            ? document.metadata.source
            : "Unknown";
        const mmrRank =
          typeof document.metadata.mmr_rank === "number"
            ? (document.metadata.mmr_rank as number)
            : idx + 1;
        const sourceColor = SOURCE_COLORS[source] ?? "rgb(107, 114, 128)";
        return (
          <article className="source-card" key={document.id}>
            <div className="source-meta">
              <span>{document.citation}</span>
              <span
                className="score-badge"
                style={{ color: scoreColor(document.score) }}
                title="Relevance score"
              >
                {document.score > 0 ? document.score.toFixed(3) : "—"}
              </span>
              <span
                style={{
                  fontSize: "0.7rem",
                  color: "rgb(148, 163, 184)",
                  fontVariantNumeric: "tabular-nums",
                }}
                title="MMR rank"
              >
                #{mmrRank}
              </span>
            </div>
            <div className="source-tags">
              <span style={{ color: sourceColor, fontWeight: 600, fontSize: "0.7rem" }}>
                {source}
              </span>
            </div>
            {document.url ? (
              <a href={document.url} target="_blank" rel="noreferrer">
                {document.title}
                <ExternalLink size={14} />
              </a>
            ) : (
              <h3>{document.title}</h3>
            )}
            <p>{document.content}</p>
          </article>
        );
      })}
    </div>
  );
});
