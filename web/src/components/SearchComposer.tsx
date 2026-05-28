import { FormEvent } from "react";
import { Loader2, Search } from "lucide-react";

interface SearchComposerProps {
  query: string;
  searchUrl: string;
  topK: number;
  isLoading: boolean;
  onQueryChange: (value: string) => void;
  onSearchUrlChange: (value: string) => void;
  onTopKChange: (value: number) => void;
  onSubmit: (event?: FormEvent) => void;
}

export function SearchComposer({
  query,
  searchUrl,
  topK,
  isLoading,
  onQueryChange,
  onSearchUrlChange,
  onTopKChange,
  onSubmit,
}: SearchComposerProps) {
  return (
    <form className="composer" onSubmit={onSubmit}>
      <textarea
        aria-label="Question"
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
        onKeyDown={(event) => {
          if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
            onSubmit();
          }
        }}
        placeholder="Ask about your indexed docs, web results, or retrieval server output"
        rows={4}
      />
      <div className="composer-controls">
        <label>
          Retrieval URL
          <input
            value={searchUrl}
            onChange={(event) => onSearchUrlChange(event.target.value)}
          />
        </label>
        <label>
          Top K
          <input
            min={1}
            max={20}
            type="number"
            value={topK}
            onChange={(event) => onTopKChange(Number(event.target.value))}
          />
        </label>
        <button type="submit" disabled={isLoading || !query.trim()}>
          {isLoading ? <Loader2 className="spin" size={18} /> : <Search size={18} />}
          <span>{isLoading ? "Searching" : "Search"}</span>
        </button>
      </div>
    </form>
  );
}
