# Design: remove the orphaned `src/internal/context/` tree

Date: 2026-07-21
Status: Approved

## Problem

`src/internal/context/` (10 files: `context/search/` + `preprocessing/` +
`retrieval/`) is onyx-heritage duplicate of the live top-level `src/context/`
retrieval/filter stack. It holds the enterprise `IndexFilters` search pipeline
(`models.py`, `pipeline.py`, `search_runner.py`, `access_filters.py`).

An import-reachability audit (graph from all production entry points, validated
against known-live modules) shows it is **unreachable from every entry point**,
and a direct grep confirms **nothing in `src` or `tests` imports it**. The live
web/search backend uses `src/context/` (`SearchFilters`), not this tree.

## Goal

Delete the dead `src/internal/context/` tree. Nothing else.

## Scope / non-goals

- Remove ONLY `src/internal/context/` (the whole directory).
- Do **not** touch `src/context/` (the live stack — different path).
- Do **not** touch the Tier-2 enterprise cluster (`document_index/weaviate`,
  `servers/indexing`, `servers/error_handling`) — those are a separate,
  documented subsystem kept for now. This means `FILTER_SEMANTICS.md` stays
  (it documents `document_index`'s `IndexFilters`, which remains).
- No behavior change; no other files edited.

## Verification

- `python -c "from src.internal.servers.web.app import create_web_app; create_web_app()"` still imports.
- Full unit suite (`pytest tests/unit`) stays green.
- `ruff check .` clean.
- Post-delete grep confirms no dangling `internal.context` references.

## Files touched

- Delete: `src/internal/context/**` (10 tracked files).
