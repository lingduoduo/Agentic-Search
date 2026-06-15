import { memo } from "react";
import type { FormEvent } from "react";
import { Loader2, Search } from "lucide-react";
import type { AgentMode } from "../types";
import type { SearchSourceProvider } from "../types";

const MODE_OPTIONS: Array<{ value: AgentMode; label: string }> = [
  { value: "search_tool", label: "Search: Direct Tool" },
  { value: "hybrid_search", label: "Search: Hybrid" },
  { value: "chat_once", label: "Chat: No Loop" },
  { value: "chat_loop", label: "Chat: Loop" },
];

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
  mode: AgentMode;
  sourceProvider: SearchSourceProvider;
  isLoading: boolean;
  onQueryChange: (value: string) => void;
  onSearchUrlChange: (value: string) => void;
  onTopKChange: (value: number) => void;
  onModeChange: (value: AgentMode) => void;
  onSourceProviderChange: (value: SearchSourceProvider) => void;
  onSubmit: (event?: FormEvent) => void;
}

export const SearchComposer = memo(function SearchComposer({
  query,
  searchUrl,
  topK,
  mode,
  sourceProvider,
  isLoading,
  onQueryChange,
  onSearchUrlChange,
  onTopKChange,
  onModeChange,
  onSourceProviderChange,
  onSubmit,
}: SearchComposerProps) {
  const isSearchMode = mode === "search_tool" || mode === "hybrid_search";
  const usesRetrievalUrl =
    !isSearchMode ||
    sourceProvider === "retrieval" ||
    sourceProvider === "browser" ||
    sourceProvider === "all";
  const urlLabel =
    sourceProvider === "browser" ? "Browser Retrieval URL" : "Retrieval URL";
  const topKLabel = isSearchMode ? "Results" : "Context Docs";

  return (
    <form className="composer" onSubmit={onSubmit}>
      <textarea
        aria-label="Question"
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
        onKeyDown={(event) => {
          if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
            event.preventDefault();
            onSubmit();
          }
        }}
        placeholder="Ask about your indexed docs, web results, or retrieval server output"
        rows={4}
      />
      <div className="composer-controls">
        <label>
          Entry Point
          <select
            value={mode}
            onChange={(event) => onModeChange(event.currentTarget.value as AgentMode)}
          >
            {MODE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        {isSearchMode && (
          <label>
            Source
            <select
              value={sourceProvider}
              onChange={(event) =>
                onSourceProviderChange(
                  event.currentTarget.value as SearchSourceProvider,
                )
              }
            >
              {SOURCE_OPTIONS.map((option) => (
                <option
                  key={option.value}
                  value={option.value}
                  disabled={option.disabled}
                  className={option.disabled ? "disabled-option" : undefined}
                >
                  {option.label}
                </option>
              ))}
            </select>
          </label>
        )}

        {usesRetrievalUrl && (
          <label className="url-field">
            {urlLabel}
            <input
              value={searchUrl}
              onChange={(event) => onSearchUrlChange(event.target.value)}
            />
          </label>
        )}

        <label>
          {topKLabel}
          <input
            min={1}
            max={20}
            type="number"
            value={topK}
            onChange={(event) => onTopKChange(event.currentTarget.valueAsNumber)}
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
