# Prune Dead Index/Document Leaf Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete three dead leaf modules (~557 LOC) and one dangling doc example from the indexing/document-store side, with zero live-behavior change.

**Architecture:** Pure removal. Each target is import-reachability-verified dead (zero non-test importers). Verification is inverted from normal TDD: instead of a failing test, each task first *proves the target is dead* via `grep`, then deletes, then proves the full suite stays green. This is PR1 of a 3-PR campaign (spec: `docs/superpowers/specs/2026-07-24-prune-dead-index-leaf-modules-design.md`).

**Tech Stack:** Python, pytest, ruff.

## Global Constraints

- Branch: `chore/prune-dead-index-leaf-modules`, already created off `origin/main`. Never commit to `main`.
- Zero change to live runtime behavior — only dead code is removed.
- `ruff check .` and `pytest` must both pass at the end of every task that deletes code.
- Do NOT touch `chunk_content_enrichment.py`, `servers/indexing/*`, workers, connectors, or any Weaviate code — those are PR2/PR3.
- Do NOT touch `embedding_cache.py` — audit traced it as live; the archived-plan deletion note is a known discrepancy, out of scope here.

---

### Task 1: Delete the dead duplicate FAISS builder (`retrieval/indexer.py`)

**Files:**
- Delete: `src/internal/retrieval/indexer.py` (115 LOC)
- Delete: `tests/unit/retrieval/test_indexer.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (removal only).

- [ ] **Step 1: Prove it's dead**

Run: `grep -rn "retrieval.indexer\|from src.internal.retrieval.indexer\|import indexer" src/ examples/ --include="*.py" | grep -v "retrieval/indexer.py"`
Expected: no output (zero non-test importers).

- [ ] **Step 2: Confirm the only test is the dedicated one**

Run: `grep -rln "retrieval.indexer\|IndexerConfig\|build_faiss_index" tests/ --include="*.py"`
Expected: only `tests/unit/retrieval/test_indexer.py`.

- [ ] **Step 3: Delete both files**

```bash
git rm src/internal/retrieval/indexer.py tests/unit/retrieval/test_indexer.py
```

- [ ] **Step 4: Verify suite is green**

Run: `ruff check . && pytest -q`
Expected: ruff passes; pytest passes (collection no longer references the deleted test).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove dead duplicate FAISS builder retrieval/indexer.py

Zero non-test importers; superseded by the live document_index build tool
(index_builder/cli/faiss_io). Removes the module + its dedicated test.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Delete orphaned Onyx batch-indexing orchestration (`document_index/indexing.py`)

**Files:**
- Delete: `src/internal/document_index/indexing.py` (402 LOC)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (removal only).

- [ ] **Step 1: Prove it's dead (no importers at all, not even tests)**

Run: `grep -rn "document_index.indexing\b\|document_index import indexing\b\|from src.internal.document_index.indexing" src/ examples/ tests/ --include="*.py"`
Expected: no output.

- [ ] **Step 2: Delete the file**

```bash
git rm src/internal/document_index/indexing.py
```

- [ ] **Step 3: Verify suite is green**

Run: `ruff check . && pytest -q`
Expected: both pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove orphaned document_index/indexing.py

Onyx batch-indexing orchestration with zero importers anywhere (not even
tests). Not on the documented index_builder build path.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Delete dead `document_metadata.py` and its smoke test

**Files:**
- Delete: `src/internal/document_index/document_metadata.py` (40 LOC)
- Modify: `tests/unit/document_index/test_imports.py` — remove `test_document_metadata_importable` only.

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (removal only).

- [ ] **Step 1: Prove the only reference is the smoke test**

Run: `grep -rn "document_metadata\|DocumentMetadata" src/ examples/ --include="*.py" | grep -v "document_metadata.py"`
Expected: no output (the `DocumentMetadata` class has no src importer).

Run: `grep -rn "document_metadata" tests/ --include="*.py"`
Expected: `tests/unit/document_index/test_imports.py` (real smoke-test import) and `tests/unit/test_db_store.py` (a test *function name* `test_connector_document_metadata_round_trips`, NOT an import — confirm by eye it is only the function name).

- [ ] **Step 2: Delete the module**

```bash
git rm src/internal/document_index/document_metadata.py
```

- [ ] **Step 3: Remove the smoke test for it**

In `tests/unit/document_index/test_imports.py`, delete exactly this block:

```python
def test_document_metadata_importable():
    import src.internal.document_index.document_metadata  # noqa: F401
```

Leave `test_chunk_content_enrichment_importable` and all others intact (that module is a PR2 target).

- [ ] **Step 4: Verify suite is green**

Run: `ruff check . && pytest -q tests/unit/document_index/test_imports.py && pytest -q`
Expected: the imports test passes without the removed case; full suite passes.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove dead document_index/document_metadata.py

DocumentMetadata has zero src importers; only reference was a smoke-test
import, removed alongside it.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Remove the dangling IVF-PQ doc example

**Files:**
- Modify: `docs/retrieval.md` (~L468-476)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (docs only).

- [ ] **Step 1: Confirm the reference is dangling**

Run: `grep -rn "index_optimizer\|FAISSIndexBuilder" src/ --include="*.py"`
Expected: no output (the module was deleted in the RRF-consolidation work; nothing replaced it).

- [ ] **Step 2: Remove the stale example block**

In `docs/retrieval.md`, delete the entire "Build an IVF-PQ FAISS index" example — the `**Build an IVF-PQ FAISS index** ...` heading line and its fenced python block that reads:

```python
from src.internal.retrieval.index_optimizer import FAISSIndexBuilder
import numpy as np

builder = FAISSIndexBuilder()
index = builder.build_ivfpq(embeddings, nlist=4096, m=96, nbits=8, nprobe=64)
# Save alongside existing index; load via FAISS_INDEX_TYPE=ivfpq
```

Remove the surrounding blank lines so the "Query transformation optimization" section that follows still reads cleanly. Do not rewrite or replace the example — there is no live equivalent.

- [ ] **Step 3: Verify no other dangling references remain in live docs**

Run: `grep -rn "index_optimizer\|FAISSIndexBuilder" docs/retrieval.md`
Expected: no output. (Archived docs under `docs/superpowers/archive/` legitimately still mention it as history — leave those.)

- [ ] **Step 4: Commit**

```bash
git add docs/retrieval.md
git commit -m "docs: drop dangling IVF-PQ example referencing deleted index_optimizer

The module was removed in the RRF-consolidation work; the example imported a
module that no longer exists.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Final full-suite gate + push + open PR

**Files:** none (verification + integration).

- [ ] **Step 1: Full lint + test gate**

Run: `ruff check . && ruff format --check . && pytest -q`
Expected: all green.

- [ ] **Step 2: Confirm the diff is deletions + one doc edit only**

Run: `git diff --stat origin/main...HEAD`
Expected: `indexer.py`, `test_indexer.py`, `indexing.py`, `document_metadata.py` deleted; `test_imports.py` and `docs/retrieval.md` trimmed; the spec + this plan added. No other source files modified.

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin chore/prune-dead-index-leaf-modules
gh pr create --base main --title "chore: prune dead index/document leaf modules (PR1 of 3)" --body "$(cat <<'EOF'
PR1 of a 3-PR dead-code-removal campaign on the indexing/document-store side.

Removes three import-reachability-verified dead leaf modules (~557 LOC) and one
dangling documentation example:

- `src/internal/retrieval/indexer.py` — dead duplicate FAISS builder (superseded by the live document_index build tool)
- `src/internal/document_index/indexing.py` — orphaned Onyx batch-indexing orchestration, zero importers
- `src/internal/document_index/document_metadata.py` — `DocumentMetadata`, zero src importers
- `docs/retrieval.md` IVF-PQ example importing the already-deleted `index_optimizer` module

Zero live-behavior change: none of these are reached by the retrieval servers
(`demo`/`hybrid`/`server`), the web backend, the example CLI, or the documented
`index_builder` build command. Only tests that exercised the deleted modules are removed.

Spec: `docs/superpowers/specs/2026-07-24-prune-dead-index-leaf-modules-design.md`
Plan: `docs/superpowers/plans/2026-07-24-prune-dead-index-leaf-modules.md`

Follow-ups: PR2 (Onyx ingestion cluster), PR3 (Weaviate). PR4 (ingestion DB tables) deferred.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR opened against `main`.

---

## Self-Review

**Spec coverage:** Every spec deletion target maps to a task — indexer.py (T1), indexing.py (T2), document_metadata.py + test_imports edit (T3), docs/retrieval.md IVF-PQ block (T4). Out-of-scope guards (enrichment, embedding_cache, Weaviate, workers) are carried in Global Constraints. Final gate + PR is T5.

**Placeholder scan:** No TBD/TODO/"handle edge cases". Every code/doc change shows exact content or exact grep/commands.

**Type consistency:** No new types introduced — removal only. Module paths used in grep steps match the delete targets exactly.
