# Move ingestion material from the README into docs/ingestion.md

**Date:** 2026-07-21
**Status:** Approved

## Problem

Continuing the per-topic README extraction (search-engine, chat-engine), the
README covers connectors, document ingestion, indexing, and background processing
only through one "What it provides" bullet. There is no dedicated page describing
the ingestion pipeline; `docs/Ingestion.md` was created empty.

## Goal

Populate `docs/ingestion.md` as a standalone overview of the ingestion pipeline —
connectors, the pre-query-time indexing jobs, and the background workers — and add
a short pointer section plus a docs-list entry in the README. No behavior changes.

## Design

1. **New `docs/ingestion.md`** (lowercase, matching all other docs; the empty
   capital-I `Ingestion.md` is removed) with a README back-link:
   - *Pipeline at a glance* — the async connectors → chunk+embed/index →
     searchable indexes flow, and that indexing happens before query time.
   - *Connectors* — `src/internal/connectors/` and the admin endpoints that manage
     connectors, credentials, OAuth, and indexing.
   - *Background processing* — the workers under
     `src/internal/servers/backgroundworker/` (beat/light/docfetching/
     docprocessing/heavy/monitoring/user-file), with index internals delegated to
     [Retrieval](../docs/retrieval.md).
   - Cross-links to `architecture.md`, `retrieval.md`, and `api-reference.md`.
2. **README edits:**
   - Add a `## Ingestion` pointer section before `## Search engine` (logical flow
     ingest → search → chat).
   - Add `docs/ingestion.md` to the Documentation list after Retrieval.
   - Keep the short "What it provides" bullet in place.

Placement is chosen to avoid colliding with the open chat-engine PR's README
edits. No code, API, or schema changes.
