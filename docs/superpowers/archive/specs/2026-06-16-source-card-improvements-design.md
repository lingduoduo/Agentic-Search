# Source Card Improvements Design Spec

**Date:** 2026-06-16
**Status:** Draft

---

## 1. Goals & Success Criteria

### Problem

`SourceGrid.tsx` renders each source document as a card with a `<p>` containing the full content text. Long documents overflow the card with no way to collapse them. There is no way to copy the content. Citation anchor IDs are absent (covered jointly with Spec 2).

### Success Criteria

- Each card's content paragraph is collapsed to 3 lines by default; a "show more ▾" button expands it per-card
- A "⎘ copy" button in the card header copies the full content to the clipboard; button text shows "copied ✓" for 1.5 s then resets
- Each card has `id="source-{citation}"` so Spec 2's citation anchor links can scroll to it
- No backend changes, no new dependencies

---

## 2. Architecture

Single-file change: `web/src/components/SourceGrid.tsx`. Each card becomes a small controlled component (`SourceCard`) that owns its own `expanded` and `copied` state. The parent `SourceGrid` remains a thin mapper.

```
SourceGrid (memo)
  └── SourceCard (memo, per document)
        ├── expanded: boolean  — controls line-clamp CSS class
        └── copied: boolean    — controls copy button label
```

---

## 3. `SourceGrid.tsx` — full rewrite of card rendering

**File:** `web/src/components/SourceGrid.tsx`

Extract a `SourceCard` sub-component above the `SourceGrid` export:

```tsx
import { memo, useState, useCallback } from "react";
import { ExternalLink } from "lucide-react";
import type { SourceDocumentView } from "../types";

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

interface SourceCardProps {
  document: SourceDocumentView;
  index: number;
}

const SourceCard = memo(function SourceCard({ document, index }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(document.content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [document.content]);

  const source =
    typeof document.metadata.source === "string"
      ? document.metadata.source
      : "Unknown";
  const mmrRank =
    typeof document.metadata.mmr_rank === "number"
      ? (document.metadata.mmr_rank as number)
      : index + 1;
  const sourceColor = SOURCE_COLORS[source] ?? "rgb(107, 114, 128)";

  return (
    <article
      className="source-card"
      key={document.id}
      id={`source-${document.citation}`}
    >
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
        <button
          className="source-copy-btn"
          onClick={handleCopy}
          title="Copy content"
          type="button"
        >
          {copied ? "copied ✓" : "⎘ copy"}
        </button>
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

      <p className={expanded ? undefined : "source-content--clamped"}>
        {document.content}
      </p>

      <button
        className="source-expand-btn"
        onClick={() => setExpanded((v) => !v)}
        type="button"
      >
        {expanded ? "show less ▴" : "show more ▾"}
      </button>
    </article>
  );
});

interface SourceGridProps {
  documents: SourceDocumentView[];
}

export const SourceGrid = memo(function SourceGrid({ documents }: SourceGridProps) {
  if (documents.length === 0) {
    return <div className="empty-state compact">No sources yet.</div>;
  }

  return (
    <div className="source-grid">
      {documents.map((document, idx) => (
        <SourceCard key={document.id} document={document} index={idx} />
      ))}
    </div>
  );
});
```

---

## 4. CSS Additions

**File:** `web/src/styles.css`

```css
/* Content clamped to 3 lines when collapsed */
.source-content--clamped {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* Copy button — inline in source-meta row */
.source-copy-btn {
  margin-left: auto;
  background: none;
  border: none;
  color: rgb(71, 85, 105);
  cursor: pointer;
  font-size: 0.65rem;
  padding: 0;
  font-family: inherit;
  transition: color 0.1s;
}
.source-copy-btn:hover { color: rgb(148, 163, 184); }

/* Expand / collapse toggle */
.source-expand-btn {
  background: none;
  border: none;
  color: rgb(56, 189, 248);
  cursor: pointer;
  font-size: 0.68rem;
  padding: 2px 0;
  margin-top: 4px;
  font-family: inherit;
  display: block;
}
.source-expand-btn:hover { text-decoration: underline; }
```

---

## 5. Error Handling

| Scenario | Behavior |
|---|---|
| `navigator.clipboard.writeText` rejects (non-HTTPS or permission denied) | `Promise` rejection ignored — button stays in default state, no crash |
| `document.content` is empty string | "show more ▾" button still renders; toggling is harmless |
| `document.citation` is empty | `id="source-"` — harmless, no citation link will target it |
| Content shorter than 3 lines | `line-clamp` has no visible effect; "show more" button still renders but does nothing visible — acceptable |

---

## 6. Testing Strategy

**File:** `web/src/components/__tests__/SourceGrid.test.tsx`

- Render one card with long content — assert `.source-content--clamped` class present
- Click "show more ▾" — assert class removed, button text changes to "show less ▴"
- Click "show less ▴" — assert class re-applied
- Mock `navigator.clipboard.writeText` — click "⎘ copy" — assert button text becomes "copied ✓"; after 1.5 s assert resets to "⎘ copy"
- Assert card has `id="source-[1]"` when `document.citation === "[1]"`

---

## 7. File Map

| Action | Path | Responsibility |
|---|---|---|
| **Modify** | `web/src/components/SourceGrid.tsx` | Extract `SourceCard`; add expand/copy/id |
| **Modify** | `web/src/styles.css` | Add `.source-content--clamped`, `.source-copy-btn`, `.source-expand-btn` |
| **Modify** | `web/src/components/__tests__/SourceGrid.test.tsx` | Expand, copy, id tests |

**Not changed:** `App.tsx`, `api.ts`, all backend files, `AnswerPanel`, `SessionTimeline`, `ToolCallTracePanel`.

**Note:** The `id="source-{citation}"` addition is required by both this spec and Spec 2 (Chat Thread + Markdown). Whichever is implemented first satisfies both; the second implementation should verify the id is already present.
