# Relabel FILTER_SEMANTICS.md Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Relabel the filter-semantics doc as the enterprise/heritage `IndexFilters` model and fix its `docs/retrieval.md` cross-reference. Docs only.

## Global Constraints

- Branch `docs/relabel-filter-semantics` (off `main`).
- Docs only; no code deletion.
- The relative link must still resolve.

---

## Task 1: Add the scope banner + fix the cross-reference

- [ ] **Step 1** — In `src/internal/document_index/FILTER_SEMANTICS.md`, replace the subtitle line ("How `IndexFilters` fields combine into the final query filter. Applies to OpenSearch.") with a scope banner:

```markdown
> **Scope: enterprise document-index backend — not the local Agentic Search stack.**
> This documents how `IndexFilters` (multi-tenant / ACL / persona / project /
> hierarchy) combine into the OpenSearch/Weaviate query filter. The running
> stack (web backend + the demo/hybrid retrieval servers) does **not** use
> `IndexFilters`; it filters with the simpler `SearchFilters`
> (`src/context/models.py`: `source_types`, `document_sets`, `tags`,
> `access_acl`, `time_cutoff`) applied post-hoc via `SearchFilters.matches()`.
> `IndexFilters` is imported only within `src/internal/document_index` and the
> `src/internal/context/search` heritage tree. Treat this file as reference for
> that enterprise index path, not live behavior.

How `IndexFilters` fields combine into the final query filter (OpenSearch/Weaviate backend).
```

- [ ] **Step 2** — In `docs/retrieval.md`, change the trailing cross-reference (currently "... `OVER_FETCH_MULTIPLIER` compensates upstream. See `src/internal/document_index/FILTER_SEMANTICS.md`.") so it reads:

```markdown
`OVER_FETCH_MULTIPLIER` compensates upstream. (The separate enterprise
document-index backend has a richer multi-tenant `IndexFilters` model — see
`src/internal/document_index/FILTER_SEMANTICS.md` — which the local stack does
not use.)
```

(Match the existing blockquote `>` prefixing if the line is inside a blockquote.)

- [ ] **Step 3** — Verify + commit:

```bash
grep -n "FILTER_SEMANTICS" docs/retrieval.md   # link still present
git diff --check
git add src/internal/document_index/FILTER_SEMANTICS.md docs/retrieval.md
git commit -m "docs: relabel FILTER_SEMANTICS.md as the enterprise index backend (not the live stack)"
```

---

## Self-Review

- **Spec coverage:** scope banner on the doc (Step 1) ✓; coherent retrieval.md cross-reference (Step 2) ✓; link resolves + clean diff (Step 3) ✓.
- **Placeholder scan:** none.
- **Accuracy:** the banner's claims (live stack uses `SearchFilters`; `IndexFilters` confined to `document_index` + `context/search`) match the verified import graph.
