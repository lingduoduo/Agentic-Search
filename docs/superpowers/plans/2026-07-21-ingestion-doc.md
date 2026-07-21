# Plan: ingestion doc extraction

**Spec:** ../specs/2026-07-21-ingestion-doc-design.md

1. Remove empty `docs/Ingestion.md`; write lowercase `docs/ingestion.md`
   (pipeline / connectors / background workers + cross-links) → verify: file
   renders, links resolve to existing docs.
2. Add a `## Ingestion` pointer section before `## Search engine` in the README →
   verify: `grep -n "ingestion.md" README.md` shows the pointer.
3. Add `docs/ingestion.md` to the README Documentation list after Retrieval →
   verify: entry present.
4. Sanity-check no dangling links → verify: `architecture.md`, `retrieval.md`,
   `api-reference.md` exist on disk.
