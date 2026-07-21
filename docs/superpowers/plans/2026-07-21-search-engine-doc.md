# Plan: search-engine doc extraction

**Spec:** ../specs/2026-07-21-search-engine-doc-design.md

1. Write `docs/search-engine.md` (capabilities summary + request-routing section
   + cross-links) → verify: file renders, links resolve to existing docs.
2. Replace README `## Request routing` (lines 66–70) with a one-line pointer →
   verify: `grep -n "search-engine.md" README.md` shows the pointer.
3. Add `docs/search-engine.md` to the README Documentation list → verify: it
   appears in the list.
4. Sanity-check no dangling links → verify: every `docs/*.md` referenced from the
   new doc exists on disk.
