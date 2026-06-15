// web/src/components/SearchComposer.tsx
import { memo } from "react";
import type { FormEvent } from "react";
import { Loader2, Search } from "lucide-react";
import type { SearchSourceProvider } from "../types";

const SOURCE_OPTIONS: Array<{
  value: SearchSourceProvider;
  label: string;
  disabled?: boolean;
}> = [
  { value: "retrieval", label: "Local Retrieval" },
  { value: "google", label: "Google PSE", disabled: true },
  { value: "serpapi", label: "SerpAPI" },
  { value: "browser", label: "Browser Retrieval" },
  { value: "all", label: "All Active Sources" },
];

interface SearchComposerProps {
  query: string;
  searchUrl: string;
  topK: number;
  sourceProvider: SearchSourceProvider;
  isLoading: boolean;
  onQueryChange: (value: string) => void;
  onSearchUrlChange: (value: string) => void;
  onTopKChange: (value: number) => void;
  onSourceProviderChange: (value: SearchSourceProvider) => void;
  onSubmit: (event?: FormEvent) => void;
}

export const SearchComposer = memo(function SearchComposer({
  query,
  searchUrl,
  topK,
  sourceProvider,
  isLoading,
  onQueryChange,
  onSearchUrlChange,
  onTopKChange,
  onSourceProviderChange,
  onSubmit,
}: SearchComposerProps) {
  return (
    <form className="composer" onSubmit={onSubmit}>
      <textarea
        aria-label="Question"
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
            e.preventDefault();
            onSubmit();
          }
        }}
        placeholder="Ask about your indexed docs, web results, or retrieval server output"
        rows={4}
      />
      <div className="composer-controls">
        <label>
          Source
          <select
            value={sourceProvider}
            onChange={(e) => onSourceProviderChange(e.currentTarget.value as SearchSourceProvider)}
          >
            {SOURCE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value} disabled={opt.disabled}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>

        <label className="url-field">
          Retrieval URL
          <input value={searchUrl} onChange={(e) => onSearchUrlChange(e.target.value)} />
        </label>

        <label>
          Top K
          <input
            min={1} max={20} type="number" value={topK}
            onChange={(e) => onTopKChange(e.currentTarget.valueAsNumber)}
          />
        </label>

        <button type="submit" disabled={isLoading || !query.trim()}>
          {isLoading ? <Loader2 className="spin" size={18} /> : <Search size={18} />}
          <span>{isLoading ? "Searching" : "Search"}</span>
        </button>
      </div>
    </form>
  );
});
