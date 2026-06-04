import { memo } from "react";
import type { FormEvent } from "react";
import { Loader2, Search } from "lucide-react";
import type { AgentMode } from "../types";

const MODE_OPTIONS: Array<{ value: AgentMode; label: string }> = [
  { value: "search_tool", label: "Search: Direct Tool" },
  { value: "hybrid_search", label: "Search: Hybrid" },
  { value: "chat_once", label: "Chat: No Loop" },
  { value: "chat_loop", label: "Chat: Loop" },
];

interface SearchComposerProps {
  query: string;
  searchUrl: string;
  topK: number;
  mode: AgentMode;
  isLoading: boolean;
  onQueryChange: (value: string) => void;
  onSearchUrlChange: (value: string) => void;
  onTopKChange: (value: number) => void;
  onModeChange: (value: AgentMode) => void;
  onSubmit: (event?: FormEvent) => void;
}

export const SearchComposer = memo(function SearchComposer({
  query,
  searchUrl,
  topK,
  mode,
  isLoading,
  onQueryChange,
  onSearchUrlChange,
  onTopKChange,
  onModeChange,
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
