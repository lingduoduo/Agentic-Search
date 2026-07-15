# Generated Context Pack

# Chat Thread + Markdown Rendering Design Spec

## Sources

- [Specification: 2026-06-16-chat-markdown-design.md](../specs/2026-06-16-chat-markdown-design.md)

## Specification Context

### 2. Architecture

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

### 9. Testing Strategy

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

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
