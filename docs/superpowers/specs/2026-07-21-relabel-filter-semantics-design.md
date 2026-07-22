# Design: relabel FILTER_SEMANTICS.md as enterprise/heritage

Date: 2026-07-21
Status: Approved

## Problem

`src/internal/document_index/FILTER_SEMANTICS.md` documents the `IndexFilters`
query-filter model (multi-tenant / ACL / persona / project / hierarchy) for the
OpenSearch/Weaviate document-index backend. That model is **not used by the live
Agentic Search retrieval path** — the running stack filters with the simpler
`SearchFilters` (`src/context/models.py`) applied post-hoc via
`SearchFilters.matches()`. `IndexFilters` is imported only within a closed
heritage loop (`src/internal/document_index` interfaces + the orphaned
`src/internal/context/search` tree, which has no live importers).

The doc reads as if it describes live behavior, and its only inbound link —
`docs/retrieval.md` — sits in a paragraph about the *live* post-hoc filter, so a
reader is pointed from live-filter prose to a different (enterprise) model.

## Goal

Relabel the doc so nobody mistakes it for live behavior, and make the
`docs/retrieval.md` cross-reference coherent — without deleting the heritage
subsystem (kept as intended future capability).

## Non-goals

- No code deletion; the `IndexFilters` / document-index / `context/search`
  subsystem stays.
- No content rewrite of the filter rules themselves (they remain accurate for
  the enterprise backend).

## Changes

1. `src/internal/document_index/FILTER_SEMANTICS.md` — add a scope banner at the
   top: this documents the **enterprise document-index backend** (`IndexFilters`,
   OpenSearch/Weaviate), which the local stack does **not** use; the live stack
   filters with `SearchFilters`.
2. `docs/retrieval.md` — reword the cross-reference so it names the doc as the
   enterprise/multi-tenant `IndexFilters` model, distinct from the live post-hoc
   `SearchFilters` just described.

## Verification

- The relative link still resolves.
- `git diff --check` clean.
