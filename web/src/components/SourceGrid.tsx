import { memo } from "react";
import { ExternalLink } from "lucide-react";
import type { SourceDocumentView } from "../types";

interface SourceGridProps {
  documents: SourceDocumentView[];
}

export const SourceGrid = memo(function SourceGrid({ documents }: SourceGridProps) {
  if (documents.length === 0) {
    return <div className="empty-state compact">No sources yet.</div>;
  }

  return (
    <div className="source-grid">
      {documents.map((document) => {
        const source =
          typeof document.metadata.source === "string"
            ? document.metadata.source
            : "Unknown";
        return (
          <article className="source-card" key={document.id}>
            <div className="source-meta">
              <span>{document.citation}</span>
              <span>{document.score.toFixed(3)}</span>
            </div>
            <div className="source-tags">
              <span>{source}</span>
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
