# Remove Orphaned `src/internal/context/` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Delete the dead `src/internal/context/` tree (10 files, zero importers) and verify nothing breaks.

## Global Constraints

- Branch `chore/remove-orphan-internal-context` (off `main`).
- Remove ONLY `src/internal/context/`. Do not touch `src/context/` or the document_index/indexing/weaviate cluster.
- The full unit suite + import smoke + ruff must stay green after deletion.

---

## Task 1: Delete the tree + verify

- [ ] **Step 1: Re-confirm zero importers**

Run: `grep -rln "internal\.context" src tests | grep -v "src/internal/context/"`
Expected: no output.

- [ ] **Step 2: Delete the directory**

```bash
git rm -r src/internal/context/
```

- [ ] **Step 3: Import smoke — the web app still builds**

Run: `python -c "from src.internal.servers.web.app import create_web_app; create_web_app()"`
Expected: no ImportError (ignore the admin-bypass log line).

- [ ] **Step 4: Full unit suite**

Run: `pytest tests/unit -q`
Expected: same pass count as `main` (no new failures; pre-existing `test_mcp_server` CORS pollution, if any, is unrelated).

- [ ] **Step 5: Lint + no dangling refs**

Run: `ruff check . && grep -rn "internal\.context" src tests | grep -v "src/internal/context/" || echo "no dangling refs"`
Expected: ruff clean; no dangling references.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: remove orphaned src/internal/context/ heritage tree

Dead onyx-heritage duplicate of the live src/context/ stack (enterprise
IndexFilters pipeline). Import-reachability audit + grep confirm zero
importers in src or tests; the live backend uses src/context/ (SearchFilters)."
```

---

## Self-Review

- **Spec coverage:** delete only `src/internal/context/` (Step 2) ✓; import smoke + full suite + ruff + dangling-ref check (Steps 3–5) ✓; no other files touched ✓.
- **Placeholder scan:** none.
- **Risk:** zero static importers confirmed twice (pre-branch + Step 1); no dynamic/factory/doc references found in the audit.
