# Generated Context Pack

# Retrieval PRD — Milestone 2: Dense Retrieval + Hybrid Fusion

## Sources

- [Plan: 2026-06-15-retrieval-m2-dense-hybrid.md](../archive/plans/2026-06-15-retrieval-m2-dense-hybrid.md)

## Implementation Plan Context

### Task 1: `fusion.py` — `rrf_fuse` and `mmr_rerank`

**Files:**
- Create: `src/internal/retrieval/fusion.py`
- Create: `tests/unit/retrieval/test_fusion.py`

- [ ] **Step 1: Write failing tests**

- [ ] **Step 2: Run tests to verify they fail**

Expected: `ModuleNotFoundError: No module named 'src.internal.retrieval.fusion'`

- [ ] **Step 3: Implement `fusion.py`**

- [ ] **Step 4: Run tests to verify they pass**

Expected: `9 passed`

- [ ] **Step 5: Commit**

---

### Task 2: Add dense leg to `LocalBackend`

**Files:**
- Modify: `src/internal/retrieval/backends/local.py`
- Modify: `tests/unit/retrieval/test_retrieval_backend.py` (append)

- [ ] **Step 1: Append failing tests**

- [ ] **Step 2: Run tests to verify they fail**

Expected: `FAILED` — `LocalBackend.__init__` does not yet accept `dense_config`

- [ ] **Step 3: Modify `local.py`**

Add `_make_dense_retriever` factory and optional dense leg to `LocalBackend.__init__` and `search_dense`:

- [ ] **Step 4: Run tests to verify they pass**

Expected: `10 passed`

- [ ] **Step 5: Commit**

---

### Task 3: Upgrade `RetrievalService.search()` to hybrid mode

**Files:**
- Modify: `src/internal/retrieval/service.py`
- Modify: `tests/unit/retrieval/test_service.py` (append)

- [ ] **Step 1: Append failing tests**

- [ ] **Step 2: Run tests to verify they fail**

Expected: `FAILED` — `search()` currently always returns `"sparse"`, no fallback logic

- [ ] **Step 3: Rewrite `service.py`**

- [ ] **Step 4: Run all service tests**

Expected: `8 passed`

- [ ] **Step 5: Commit**

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
