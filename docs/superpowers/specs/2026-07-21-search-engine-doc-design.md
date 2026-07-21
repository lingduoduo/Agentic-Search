# Move search-agent material from the README into docs/search-engine.md

**Date:** 2026-07-21
**Status:** Approved

## Problem

The top-level `README.md` carries search-engine detail that bloats the landing
page and belongs in a dedicated guide:

- The **Request routing** section (README lines 66–70) — the routing heading plus
  two dense paragraphs on auto-routed search, the SerpAPI/browser fallthrough,
  the strong-unfiltered direct-first path, and the shared composed pipeline.
- Search-specific capability bullets in **What it provides** (agentic RAG /
  multi-turn search, dense-sparse-hybrid retrieval, web search).

`docs/search-engine.md` exists but is empty. There is no single page describing
the search agent end to end; the material is scattered between the README and the
existing `docs/request-routing.md` and `docs/retrieval.md`.

## Goal

Populate `docs/search-engine.md` as a standalone overview of the search agent —
its capabilities and its request-routing behavior — and slim the README to short
bullets plus a one-line pointer, without changing any product behavior.

## Design

1. **New `docs/search-engine.md`** with a back-link to the README:
   - *Capabilities* — a short summary drawn from the search-related "What it
     provides" bullets (agentic RAG / multi-turn search, dense/sparse/hybrid
     retrieval, web search).
   - *Request routing* — the full content of README lines 66–70, verbatim in
     substance, with links preserved to `docs/request-routing.md`.
   - Cross-links to `docs/request-routing.md` and `docs/retrieval.md` for the
     authoritative deep dives (no duplication of their internals).
2. **README edits:**
   - Replace the `## Request routing` section (lines 66–70) with a one-line
     pointer to `docs/search-engine.md`.
   - Keep the short capability bullets in place (already concise); the new doc
     summarizes them.
   - Add `docs/search-engine.md` to the Documentation list.

No code, API, or schema changes.
