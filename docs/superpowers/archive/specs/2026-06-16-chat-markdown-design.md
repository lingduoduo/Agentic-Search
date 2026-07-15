# Chat Thread + Markdown Rendering Design Spec

**Date:** 2026-06-16
**Status:** Draft

---

## 1. Goals & Success Criteria

### Problem

- `AnswerPanel` splits the answer on `\n\n` and renders raw `<p>` tags — bold, lists, and code blocks from the LLM are displayed as literal asterisks and backticks.
- Citation references (`[1]`, `[2]`) in the answer text are inert `<span>` elements in a separate `.citation-row` div — not linked to the source cards they reference.
- `SessionTimeline` renders all messages (including `system`) as a plain numbered `<ol>` with `<strong>role</strong>` labels — no visual distinction between user and assistant turns.

### Success Criteria

- LLM answer text renders as Markdown: bold, italics, bullet lists, numbered lists, inline code, and fenced code blocks all display correctly
- `[n]` citation patterns in the answer text become anchor links; clicking one scrolls the matching source card into view
- `SessionTimeline` displays user messages as left-aligned bubbles and assistant messages as right-aligned bubbles; system messages are hidden
- `rounds_used` / `num_turns` metadata remains visible below assistant bubbles
- No new backend changes required — all three changes are frontend-only

---

## 2. Architecture

Three self-contained changes, each in a single component file:

```
AnswerPanel.tsx   ←  react-markdown replaces paragraph splitter
                      custom text renderer injects citation anchor links
SessionTimeline.tsx ← <ol> replaced with flex bubble layout
                        system messages filtered out
SourceGrid.tsx    ←  id="source-{citation}" added to each <article>
```

Citation scroll flow:
```
User clicks [1] in AnswerPanel
  → browser navigates to #source-[1]
  → SourceGrid article with id="source-[1]" scrolls into view
```

---

## 3. Dependency

**File:** `web/package.json`

Add `react-markdown` (compatible with React 19):

```json
"react-markdown": "^9.0.1"
```

`react-markdown` v9 is ESM-only and requires no additional peer dependencies for the features used here. No `remark-*` or `rehype-*` plugins needed — the default renderer handles bold, italic, lists, and code blocks out of the box.

Install: `npm install react-markdown`

---

## 4. AnswerPanel — Markdown + Citation Links

**File:** `web/src/components/AnswerPanel.tsx`

### 4.1 Replace paragraph splitter with ReactMarkdown

Remove:
```tsx
const paragraphs = useMemo(() => answer.split(/\n\n+/).filter(Boolean), [answer]);
// ...
{paragraphs.map((para, i) => (
  <p key={i}>{para}</p>
))}
```

Replace with:
```tsx
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";

// ...

<ReactMarkdown components={markdownComponents}>{answer}</ReactMarkdown>
```

### 4.2 Citation link renderer

Define `markdownComponents` to intercept plain text nodes and replace `[n]` patterns with anchor links:

```tsx
const CITATION_RE = /(\[\d+\])/g;

const markdownComponents: Components = {
  // Intercept text inside paragraphs to linkify [n] patterns
  p({ children }) {
    return (
      <p>
        {linkifyCitations(children)}
      </p>
    );
  },
  // Prevent react-markdown from wrapping code blocks in extra divs
  code({ className, children }) {
    const isBlock = className?.startsWith("language-");
    return isBlock ? (
      <pre><code className={className}>{children}</code></pre>
    ) : (
      <code>{children}</code>
    );
  },
};

function linkifyCitations(children: React.ReactNode): React.ReactNode {
  // react-markdown passes paragraph children as an array of strings/elements
  if (Array.isArray(children)) {
    return children.map((child, i) => (
      <React.Fragment key={i}>{linkifyCitations(child)}</React.Fragment>
    ));
  }
  if (typeof children !== "string") return children;
  const parts = children.split(CITATION_RE);
  if (parts.length === 1) return children; // no [n] pattern — fast path
  return parts.map((part, i) =>
    CITATION_RE.test(part) ? (
      <a
        key={i}
        href={`#source-${part}`}
        className="citation-link"
        onClick={(e) => {
          e.preventDefault();
          document.getElementById(`source-${part}`)?.scrollIntoView({
            behavior: "smooth",
            block: "nearest",
          });
        }}
      >
        {part}
      </a>
    ) : (
      part
    )
  );
}
```

### 4.3 Remove the citation-row div

The bottom citation row (`<div className="citation-row">`) is replaced by inline links in the answer text. Remove it entirely.

---

## 5. SessionTimeline — Chat Bubble Layout

**File:** `web/src/components/SessionTimeline.tsx`

Replace the entire `<ol>` rendering with a flex-column bubble layout. Filter out `system` role messages.

```tsx
export const SessionTimeline = memo(function SessionTimeline({
  messages,
}: SessionTimelineProps) {
  const visible = messages.filter((m) => m.role !== "system");

  if (visible.length === 0) {
    return <div className="empty-state compact">Start a query to create history.</div>;
  }

  return (
    <div className="chat-thread">
      {visible.map((message, index) => {
        const isUser = message.role === "user";
        const roundsUsed = message.metadata?.rounds_used as number | undefined;
        const numTurns = message.metadata?.num_turns as number | undefined;

        return (
          <div
            key={`${message.role}-${index}`}
            className={`chat-row ${isUser ? "chat-row--user" : "chat-row--assistant"}`}
          >
            {isUser && <div className="chat-avatar chat-avatar--user">U</div>}

            <div className="chat-bubble-wrapper">
              <div className={`chat-bubble ${isUser ? "chat-bubble--user" : "chat-bubble--assistant"}`}>
                {message.content}
              </div>
              {!isUser && (roundsUsed != null || numTurns != null) && (
                <div className="chat-meta">
                  {roundsUsed != null && `${roundsUsed} rounds`}
                  {roundsUsed != null && numTurns != null && " · "}
                  {numTurns != null && `${numTurns} turns`}
                </div>
              )}
            </div>

            {!isUser && <div className="chat-avatar chat-avatar--assistant">A</div>}
          </div>
        );
      })}
    </div>
  );
});
```

---

## 6. SourceGrid — Citation Anchor IDs

**File:** `web/src/components/SourceGrid.tsx`

Add `id` to each source card `<article>` so citation links can scroll to them:

```tsx
// Before
<article className="source-card" key={document.id}>

// After
<article
  className="source-card"
  key={document.id}
  id={`source-${document.citation}`}
>
```

`document.citation` is already a string like `"[1]"`, so `id="source-[1]"` matches the `href="#source-[1]"` link in AnswerPanel exactly. Square brackets are valid in HTML `id` attributes.

---

## 7. CSS Additions

**File:** `web/src/styles.css`

```css
/* Citation links in answer text */
.citation-link {
  color: var(--color-accent, #38bdf8);
  background: rgba(56, 189, 248, 0.1);
  padding: 1px 5px;
  border-radius: 3px;
  text-decoration: none;
  font-size: 0.8em;
  font-variant-numeric: tabular-nums;
}
.citation-link:hover {
  background: rgba(56, 189, 248, 0.2);
  text-decoration: underline;
}

/* Chat thread container */
.chat-thread {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 4px 0;
}

/* Chat row: user left, assistant right */
.chat-row {
  display: flex;
  align-items: flex-start;
  gap: 8px;
}
.chat-row--user  { justify-content: flex-start; }
.chat-row--assistant { justify-content: flex-end; }

/* Avatar circle */
.chat-avatar {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.6rem;
  font-weight: 600;
}
.chat-avatar--user        { background: #3b82f6; color: #fff; }
.chat-avatar--assistant   { background: #1e3a5f; color: #38bdf8; }

/* Bubble wrapper holds bubble + meta line */
.chat-bubble-wrapper {
  max-width: 85%;
  display: flex;
  flex-direction: column;
}

/* Speech bubble */
.chat-bubble {
  padding: 7px 11px;
  border-radius: 8px;
  font-size: 0.8rem;
  line-height: 1.5;
  word-break: break-word;
}
.chat-bubble--user {
  background: #1e293b;
  border-radius: 2px 8px 8px 8px;
  color: #e2e8f0;
}
.chat-bubble--assistant {
  background: #0d1f33;
  border: 1px solid #1e3a5f;
  border-radius: 8px 2px 8px 8px;
  color: #e2e8f0;
}

/* rounds/turns metadata under assistant bubble */
.chat-meta {
  font-size: 0.62rem;
  color: #475569;
  margin-top: 3px;
  text-align: right;
  padding-right: 2px;
}
```

---

## 8. Error Handling

| Scenario | Behavior |
|---|---|
| Answer contains no `[n]` citations | `linkifyCitations` returns text unchanged — no links rendered |
| `document.citation` is empty string | `id="source-"` — harmless, no matching `href` in AnswerPanel |
| Citation link clicked but card not in DOM | `getElementById` returns `null`; `?.scrollIntoView()` is a no-op |
| Answer contains fenced code block | `react-markdown` renders `<pre><code>` correctly via the custom `code` component |

---

## 9. Testing Strategy

- **`web/src/components/__tests__/AnswerPanel.test.tsx`**
  - Render with answer `"See **bold** and [1] for more"` — assert `<strong>` tag present, `<a href="#source-[1]">` present
  - Render with no citations in text — assert no `<a>` rendered
  - Assert `.citation-row` div is gone

- **`web/src/components/__tests__/SessionTimeline.test.tsx`**
  - Render with `[{role:"system",...}, {role:"user",...}, {role:"assistant",...}]` — assert system message absent, two bubbles present
  - Assert user bubble has `chat-row--user` class, assistant has `chat-row--assistant`
  - Assert `rounds_used` / `num_turns` rendered in `.chat-meta` when present in metadata

- **`web/src/components/__tests__/SourceGrid.test.tsx`**
  - Render with one document (`citation: "[1]"`) — assert article has `id="source-[1]"`

---

## 10. File Map

| Action | Path | Responsibility |
|---|---|---|
| **Modify** | `web/package.json` | Add `react-markdown ^9.0.1` |
| **Modify** | `web/src/components/AnswerPanel.tsx` | ReactMarkdown + `linkifyCitations` + remove citation-row |
| **Modify** | `web/src/components/SessionTimeline.tsx` | Chat bubble layout; filter system messages |
| **Modify** | `web/src/components/SourceGrid.tsx` | Add `id="source-{citation}"` to each article |
| **Modify** | `web/src/styles.css` | Add citation-link and chat-thread CSS |
| **Modify** | `web/src/components/__tests__/AnswerPanel.test.tsx` | Markdown + citation link tests |
| **Modify** | `web/src/components/__tests__/SessionTimeline.test.tsx` | Bubble layout + system filter tests |
| **Modify** | `web/src/components/__tests__/SourceGrid.test.tsx` | Citation anchor id test |

**Not changed:** `App.tsx`, `api.ts`, all backend files, `SearchComposer`, `ToolPanel`, `ConnectorPanel`.
