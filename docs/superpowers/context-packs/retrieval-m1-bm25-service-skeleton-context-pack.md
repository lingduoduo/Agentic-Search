# Generated Context Pack

# Retrieval PRD — Milestone 1: BM25 Baseline + Service Skeleton

## Sources

- [Plan: 2026-06-15-retrieval-m1-bm25-service-skeleton.md](../archive/plans/2026-06-15-retrieval-m1-bm25-service-skeleton.md)

## Implementation Plan Context

### Task 1: `RetrievalResult` dataclass + `RetrievalBackend` ABC

**Files:**
- Create: `src/internal/retrieval/__init__.py`
- Create: `src/internal/retrieval/backends/__init__.py`
- Create: `src/internal/retrieval/backends/base.py`
- Test: `tests/unit/retrieval/test_retrieval_backend.py`

- [ ] **Step 1: Write failing tests**

- [ ] **Step 2: Run tests to verify they fail**

Expected: `ModuleNotFoundError: No module named 'src.internal.retrieval'`

- [ ] **Step 3: Create package markers**

```python
# src/internal/retrieval/backends/__init__.py
python
# src/internal/retrieval/backends/base.py
"""Abstract base for all retrieval backends."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

@dataclass

…

### Task 2: `LocalBackend` wrapping `SparseRetriever`

**Files:**
- Create: `src/internal/retrieval/backends/local.py`
- Modify: `tests/unit/retrieval/test_retrieval_backend.py` (append)

- [ ] **Step 1: Append failing tests**

- [ ] **Step 2: Run tests to verify they fail**

Expected: `ImportError` on `local.py` imports

- [ ] **Step 3: Implement `local.py`**

- [ ] **Step 4: Run tests to verify they pass**

Expected: `8 passed`

- [ ] **Step 5: Commit**

---

### Task 3: `RetrievalService` with backend selection

**Files:**
- Create: `src/internal/retrieval/service.py`
- Create: `tests/unit/retrieval/test_service.py`

- [ ] **Step 1: Write failing tests**

- [ ] **Step 2: Run tests to verify they fail**

Expected: `ModuleNotFoundError: No module named 'src.internal.retrieval.service'`

- [ ] **Step 3: Implement `service.py`**

- [ ] **Step 4: Run tests to verify they pass**

Expected: `4 passed`

- [ ] **Step 5: Commit**

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
