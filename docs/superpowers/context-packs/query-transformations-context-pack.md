# Generated Context Pack

# Query Transformations

## Sources

- [Specification: 2026-06-16-query-transformations-design.md](../archive/specs/2026-06-16-query-transformations-design.md)
- [Plan: 2026-06-16-query-transformations.md](../archive/plans/2026-06-16-query-transformations.md)

## Specification Context

### Out of Scope

- Replacing `QueryEnhancer` or `AgenticRAGLoop` (they continue to work as-is)
- HTTP endpoint for query transformation
- Streaming transformed results
- Training or fine-tuning query transformation models
- Query routing to different HTTP retrieval servers (only backend-strategy routing via filters)

---

### 2. Architecture

When `pipeline` is `None` (default), `RetrievalService.search()` behaves exactly as today — single query, no fusion step, no extra latency.

---

### `test_query_transform.py`

- Pipeline with all flags off → `TransformedQueryBundle` with only `original`, all lists empty
- `retrieval_variants()` deduplicates case-insensitively and respects `max_variants`
- `retrieval_variants()` always includes `original` even when all transformers return empty
- Each `QT_*` flag independently enables its transformer (monkeypatch `QueryEnhancer`)
- `from_env()` returns `None` when no `QT_*` vars set
- `from_env()` returns pipeline when at least one `QT_*` var is `true`

## Implementation Plan Context

### Task 1: `TransformedQueryBundle` + `QueryTransformConfig`

**Files:**
- Create: `src/context/query_transform.py`
- Create: `tests/unit/test_query_transform.py`

- [ ] **Step 1: Write the failing tests**

- [ ] **Step 2: Run tests to confirm they fail**

Expected: `ModuleNotFoundError: No module named 'src.context.query_transform'`

- [ ] **Step 3: Write the implementation**

- [ ] **Step 4: Run tests to confirm they pass**

Expected: `5 passed`

- [ ] **Step 5: Commit**

---

### Task 2: `QueryConstructor`

**Files:**
- Create: `src/internal/retrieval/query_constructor.py`
- Create: `tests/unit/retrieval/test_query_constructor.py`

- [ ] **Step 1: Write the failing tests**

- [ ] **Step 2: Run tests to confirm they fail**

Expected: `ModuleNotFoundError: No module named 'src.internal.retrieval.query_constructor'`

- [ ] **Step 3: Write the implementation**

- [ ] **Step 4: Run tests to confirm they pass**

Expected: `6 passed`

- [ ] **Step 5: Commit**

---

### Task 3: `QueryTransformPipeline`

**Files:**
- Modify: `src/context/query_transform.py` (append class)
- Modify: `tests/unit/test_query_transform.py` (append tests)

- [ ] **Step 1: Write the failing tests — append to existing test file**

- [ ] **Step 2: Run tests to confirm they fail**

Expected: failures on the new tests (ImportError for `QueryTransformPipeline`)

- [ ] **Step 3: Write the implementation — append to `src/context/query_transform.py`**

Add these imports at the top of the existing file (after `import os`):

Then append at the bottom of `src/context/query_transform.py`:

- [ ] **Step 4: Run all query_transform tests**

Expected: `13 passed`

- [ ] **Step 5: Commit**

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
