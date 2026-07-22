# Spec: Data-source cards must never render "Unknown"

## Problem

Result / citation cards in the web UI display the literal string **"Unknown"**
as the data source for some answers. This happens for the classic
retrieval-augmented answer path (`/api/agent` → `answer_with_retrieval`).

### Root cause

The local retrieval servers (`demo.py`, `hybrid.py`) return each document as
`{id, title, text, url}` with **no `source` field** in its metadata. The web
backend attaches a human-readable provider label
(`metadata["source"] = "Local Retrieval"`) via `_document_with_metadata()` on
most paths — search-agent, agentic-RAG, direct search, auto-route. But the
**classic `answer_with_retrieval` path** (`app.py`, the fallback branch of
`_run_agent_impl`) passes `result.context.documents` straight to
`_finalize_response` with no label. The frontend then falls back to the literal
`"Unknown"` (`SourceGrid.tsx`).

## Goal

The source tag on every result card shows a real **provider label** (e.g.
"Local Retrieval"), never "Unknown".

## Approach (chosen)

Provider-level label (not per-document origin). Two layers:

1. **Backend (root cause):** stamp the provider label on the classic
   `answer_with_retrieval` path, matching the convention already used by the
   other loop runners (`source_provider="retrieval"` → "Local Retrieval").
2. **Frontend (defense in depth):** the `SourceGrid` fallback no longer emits
   "Unknown"; a document whose metadata lacks a `source` defaults to
   "Local Retrieval" (the app's default retrieval backend).

## Follow-up (same PR): per-document real source

After the provider-label fix landed, we extended the change so cards show each
document's **real origin** rather than a uniform provider label:

- Documents carry a per-document `metadata.source`. Precedence:
  explicit per-doc `source` > per-corpus default (`corpora.json` `source`
  field) > provider label ("Local Retrieval").
- The corpus registry (`resolve_corpus_docs`) stamps the source at load time.
- The retrieval servers (`demo.py`, `hybrid.py`) forward `metadata` on
  `/retrieve` (they previously dropped it).
- The web backend (`_document_with_metadata`) prefers the document's real
  `source`, falling back to the provider label only when absent.

The provider label remains the fallback, so nothing ever renders "Unknown".

## Non-goals

- Changing the provider label set or per-provider colors.
- Per-source colors for the new corpus sources (they use the default swatch).

## Acceptance criteria

- The classic `answer_with_retrieval` path returns documents whose
  `metadata["source"] == "Local Retrieval"`.
- `SourceGrid.tsx` never renders the string "Unknown".
- A regression test covers the classic-RAG source label.
- `pytest` + frontend `npm run typecheck` pass.
