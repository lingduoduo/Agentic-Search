# Design: documentation sync for the direct surfaces

Date: 2026-07-21
Status: Approved

## Problem

Four merged PRs this session added user-facing behavior that the docs do not yet
fully reflect:

- #448 — three direct surfaces (Search/Chat/Tool) + `POST /chat/send-chat-message`
  + the `web_search` serpapi→browser cascade.
- #449 — a running transcript in the Chat and Tool Agent surfaces.
- #450 — tool-approval parity on the Tool Agent surface.
- #451 — visual polish for the four-tab surface.

Current doc gaps:
- `docs/search-engine.md` documents only the auto-router search — no mention of
  the dedicated `POST /search/send-search-message` surface or the Search tab.
- `docs/chat-engine.md` has the direct chat endpoint but not the transcript.
- `docs/tool-engine.md` has the tool surface, cascade, and approvals but the UI
  note predates the transcript + inline approval prompts.
- `README.md` describes the engines and the frontend but not the four dedicated
  surfaces (Assistant | Search | Chat | Tool Agent) or the transcript.

## Goal

Sync the four docs to the merged, verifiable behavior. Documentation only — no
code changes.

## Non-goals (YAGNI)

- No code changes; no new claims beyond what shipped.
- No restructuring of the docs; additive edits placed in the existing sections.
- Do not document unmerged or speculative behavior.

## Changes (all verified against merged `main`)

### `docs/search-engine.md`
Add a "Dedicated search surface (`/search/*`)" section paralleling the tool-engine
doc: `POST /search/send-search-message` (retrieval only — returns docs, no LLM
synthesis; JSON, or NDJSON when `stream:true`), `GET /search/search-history`, and
that the web UI **Search** tab drives it.

### `docs/chat-engine.md`
Extend the direct-chat section: the **Chat** tab renders a running transcript of
the session's turns (client-side accumulation).

### `docs/tool-engine.md`
Update the UI note: the **Tool Agent** tab shows a running transcript that
interleaves each turn's tool-call trace, and renders inline approval prompts for
gated tools.

### `README.md`
- "What it provides": update the UI bullet to name the four surfaces
  (Assistant auto-router + direct Search / Chat / Tool Agent) and the transcript.
- Search / Chat / Tool engine sections: one line each naming the dedicated
  surface + endpoint.
- "Start the frontend": note the four-tab switcher.

## Verification

- Every endpoint/behavior named is checked against merged `main`
  (`/search/*`, `/chat/send-chat-message`, `/tool/*`, the cascade, approvals).
- Relative Markdown links resolve.

## Files touched

- `docs/search-engine.md`, `docs/chat-engine.md`, `docs/tool-engine.md`,
  `README.md`.
