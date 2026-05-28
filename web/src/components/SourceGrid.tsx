import { ExternalLink } from "lucide-react";
import type { SourceDocumentView } from "../types";

interface SourceGridProps {
  documents: SourceDocumentView[];
}

export function SourceGrid({ documents }: SourceGridProps) {
  if (documents.length === 0) {
    return <div className="empty-state compact">No sources yet.</div>;
  }

  return (
    <div className="source-grid">
      {documents.map((document) => (
        <article className="source-card" key={document.id}>
          <div className="source-meta">
            <span>{document.citation}</span>
            <span>{document.score.toFixed(3)}</span>
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
      ))}
    </div>
  );
}
