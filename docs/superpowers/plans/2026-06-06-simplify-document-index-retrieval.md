# Simplify document_index + retrieval Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate backward-compat shims, rename a misnamed interface file, and replace a no-op timing decorator stub with a real implementation across `src/backend/document_index/` and `src/retrieval/`.

**Architecture:** Three focused changes with zero behavior-breaking impact: (1) delete the 15-line `vector_db_insertion.py` shim that no caller imports; (2) rename `interfaces_new.py` to `interfaces.py` and update the 8 files that import it; (3) swap the `log_function_time` no-op decorator in `utils.py` for a real `time.perf_counter` timer so the 25+ decorated methods in `client.py` actually log timing at DEBUG level. `text_processor.py` stays in `src/retrieval/` — it's correctly integrated there via `vocabulary.py` and is unrelated to the OpenSearch indexing pipeline.

**Tech Stack:** Python 3.11, pytest, ruff, OpenSearch Python client

---

## File Map

| Change | File |
|--------|------|
| Delete | `src/retrieval/vector_db_insertion.py` |
| Modify | `src/backend/document_index/indexing.py` — remove compat alias + `__all__` entry |
| Rename (git mv) | `src/backend/document_index/interfaces_new.py` → `interfaces.py` |
| Update imports | `src/backend/document_index/disabled.py` |
| Update imports | `src/backend/document_index/factory.py` |
| Update imports | `src/backend/document_index/opensearch/client.py` |
| Update imports | `src/backend/document_index/opensearch/opensearch_document_index.py` |
| Update imports | `src/backend/document_index/opensearch/schema.py` |
| Update imports | `src/backend/document_index/opensearch/search.py` |
| Update test | `tests/unit/document_index/test_imports.py` |
| Update test | `tests/unit/document_index/test_disabled.py` |
| Modify | `src/backend/document_index/utils.py` — replace `log_function_time` no-op with real timer |

---

## Task 1: Delete `vector_db_insertion.py` and the compat alias in `indexing.py`

**Context:** `src/retrieval/vector_db_insertion.py` is a 15-line shim that re-exports `ChunkSink`, `write_chunks_with_backoff`, and an alias `write_chunks_to_vector_db_with_backoff` from `src/backend/document_index/indexing.py`. A full codebase grep confirms zero files import from this shim. The alias also exists inside `indexing.py` itself (line 386) and is listed in `__all__`. Both the shim file and the alias are dead code.

**Files:**
- Delete: `src/retrieval/vector_db_insertion.py`
- Modify: `src/backend/document_index/indexing.py` (lines 386, 388-403)

- [ ] **Step 1: Confirm no imports of `vector_db_insertion`**

```bash
grep -rn "vector_db_insertion\|write_chunks_to_vector_db_with_backoff" \
  --include="*.py" | grep -v "__pycache__" \
  | grep -v "src/retrieval/vector_db_insertion.py" \
  | grep -v "src/backend/document_index/indexing.py"
```

Expected: no output. If any lines appear, update those callers to import from `src.backend.document_index.indexing` before continuing.

- [ ] **Step 2: Delete the shim file**

```bash
git rm src/retrieval/vector_db_insertion.py
```

- [ ] **Step 3: Remove the compat alias and `__all__` entry from `indexing.py`**

In `src/backend/document_index/indexing.py`, replace:

```python
write_chunks_to_vector_db_with_backoff = write_chunks_with_backoff

__all__ = [
    "ChunkBatchStore",
    "ChunkSink",
    "Chunker",
    "DefaultIndexingEmbedder",
    "DocumentBatchPrepareContext",
    "DocumentIndexingResult",
    "IndexingEmbedder",
    "embed_and_stream",
    "filter_documents",
    "index_document_batch",
    "index_documents",
    "numpy_embedding_fn",
    "write_chunks_to_vector_db_with_backoff",
    "write_chunks_with_backoff",
]
```

with:

```python
__all__ = [
    "ChunkBatchStore",
    "ChunkSink",
    "Chunker",
    "DefaultIndexingEmbedder",
    "DocumentBatchPrepareContext",
    "DocumentIndexingResult",
    "IndexingEmbedder",
    "embed_and_stream",
    "filter_documents",
    "index_document_batch",
    "index_documents",
    "numpy_embedding_fn",
    "write_chunks_with_backoff",
]
```

- [ ] **Step 4: Run linter**

```bash
ruff check src/retrieval/ src/backend/document_index/indexing.py --fix && ruff format src/backend/document_index/indexing.py
```

Expected: no errors.

- [ ] **Step 5: Run tests**

```bash
pytest tests/unit/ -v -x
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/backend/document_index/indexing.py
git commit -m "chore: delete vector_db_insertion compat shim and alias"
```

---

## Task 2: Rename `interfaces_new.py` → `interfaces.py` and update all import sites

**Context:** The file is named `interfaces_new.py` — a temporary suffix from when it replaced an older interface file. The "old" file is long gone, so the `_new` suffix is misleading. Eight files import from `interfaces_new`; all must be updated atomically in one commit.

**Files:**
- Rename: `src/backend/document_index/interfaces_new.py` → `src/backend/document_index/interfaces.py`
- Update imports in: `disabled.py`, `factory.py`, `opensearch/client.py`, `opensearch/opensearch_document_index.py`, `opensearch/schema.py`, `opensearch/search.py`
- Update tests in: `tests/unit/document_index/test_imports.py`, `tests/unit/document_index/test_disabled.py`

- [ ] **Step 1: Rename the file**

```bash
git mv src/backend/document_index/interfaces_new.py src/backend/document_index/interfaces.py
```

- [ ] **Step 2: Update `disabled.py`**

Replace:
```python
from src.backend.document_index.interfaces_new import DocumentIndex
from src.backend.document_index.interfaces_new import DocumentInsertionRecord
from src.backend.document_index.interfaces_new import DocumentSectionRequest
from src.backend.document_index.interfaces_new import IndexingMetadata
from src.backend.document_index.interfaces_new import MetadataUpdateRequest
```
with:
```python
from src.backend.document_index.interfaces import DocumentIndex
from src.backend.document_index.interfaces import DocumentInsertionRecord
from src.backend.document_index.interfaces import DocumentSectionRequest
from src.backend.document_index.interfaces import IndexingMetadata
from src.backend.document_index.interfaces import MetadataUpdateRequest
```

- [ ] **Step 3: Update `factory.py`**

Replace:
```python
from src.backend.document_index.interfaces_new import DocumentIndex
```
with:
```python
from src.backend.document_index.interfaces import DocumentIndex
```

And the deferred import inside `_build_tenant_state()`:
```python
    from src.backend.document_index.interfaces_new import TenantState
```
with:
```python
    from src.backend.document_index.interfaces import TenantState
```

- [ ] **Step 4: Update `opensearch/client.py`**

Replace:
```python
from src.backend.document_index.interfaces_new import TenantState
```
with:
```python
from src.backend.document_index.interfaces import TenantState
```

- [ ] **Step 5: Update `opensearch/opensearch_document_index.py`**

Replace:
```python
from src.backend.document_index.interfaces_new import DocumentIndex
from src.backend.document_index.interfaces_new import DocumentInsertionRecord
from src.backend.document_index.interfaces_new import DocumentSectionRequest
from src.backend.document_index.interfaces_new import IndexingMetadata
from src.backend.document_index.interfaces_new import MetadataUpdateRequest
from src.backend.document_index.interfaces_new import TenantState
```
with:
```python
from src.backend.document_index.interfaces import DocumentIndex
from src.backend.document_index.interfaces import DocumentInsertionRecord
from src.backend.document_index.interfaces import DocumentSectionRequest
from src.backend.document_index.interfaces import IndexingMetadata
from src.backend.document_index.interfaces import MetadataUpdateRequest
from src.backend.document_index.interfaces import TenantState
```

- [ ] **Step 6: Update `opensearch/schema.py`**

Replace:
```python
from src.backend.document_index.interfaces_new import TenantState
```
with:
```python
from src.backend.document_index.interfaces import TenantState
```

- [ ] **Step 7: Update `opensearch/search.py`**

Replace:
```python
from src.backend.document_index.interfaces_new import TenantState
```
with:
```python
from src.backend.document_index.interfaces import TenantState
```

- [ ] **Step 8: Update `tests/unit/document_index/test_imports.py`**

Replace:
```python
def test_interfaces_new_importable():
    import src.backend.document_index.interfaces_new  # noqa: F401
```
with:
```python
def test_interfaces_importable():
    import src.backend.document_index.interfaces  # noqa: F401
```

- [ ] **Step 9: Update `tests/unit/document_index/test_disabled.py`**

Replace:
```python
from src.backend.document_index.interfaces_new import (
    IndexingMetadata,
)
```
with:
```python
from src.backend.document_index.interfaces import (
    IndexingMetadata,
)
```

- [ ] **Step 10: Verify no `interfaces_new` references remain**

```bash
grep -rn "interfaces_new" --include="*.py" | grep -v "__pycache__"
```

Expected: no output.

- [ ] **Step 11: Run linter**

```bash
ruff check src/backend/document_index/ tests/unit/document_index/ --fix && \
ruff format src/backend/document_index/ tests/unit/document_index/
```

Expected: no errors.

- [ ] **Step 12: Run tests**

```bash
pytest tests/unit/ -v -x
```

Expected: all pass.

- [ ] **Step 13: Commit**

```bash
git add src/backend/document_index/interfaces.py \
  src/backend/document_index/disabled.py \
  src/backend/document_index/factory.py \
  src/backend/document_index/opensearch/client.py \
  src/backend/document_index/opensearch/opensearch_document_index.py \
  src/backend/document_index/opensearch/schema.py \
  src/backend/document_index/opensearch/search.py \
  tests/unit/document_index/test_imports.py \
  tests/unit/document_index/test_disabled.py
git commit -m "refactor: rename interfaces_new.py to interfaces.py"
```

---

## Task 3: Replace `log_function_time` no-op stub with real timer

**Context:** `src/backend/document_index/utils.py` defines `log_function_time` as a no-op decorator factory (it wraps functions but does nothing). This decorator is applied to 25+ methods in `opensearch/client.py`. Making it a real timer (logging elapsed time at DEBUG level) costs zero import changes and gives operators actual timing data when they set `DEBUG` logging.

The existing signature is `log_function_time(*, print_only, debug_only, include_args, include_args_subset)`. The replacement must match this signature exactly so `client.py` call sites need no changes.

**Files:**
- Modify: `src/backend/document_index/utils.py`
- Modify: `tests/unit/document_index/test_types.py` (update test to assert timing is logged)

- [ ] **Step 1: Write failing test**

In `tests/unit/document_index/test_types.py`, add a new test:

```python
def test_log_function_time_logs_timing(caplog):
    import logging
    from src.backend.document_index.utils import log_function_time

    @log_function_time(debug_only=True)
    def slow_fn():
        return 42

    with caplog.at_level(logging.DEBUG):
        result = slow_fn()

    assert result == 42
    assert any("slow_fn" in r.message and "took" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/document_index/test_types.py::test_log_function_time_logs_timing -v
```

Expected: FAIL — `assert any(...)` is False because the no-op logs nothing.

- [ ] **Step 3: Replace the `log_function_time` stub in `utils.py`**

In `src/backend/document_index/utils.py`, find and replace the `log_function_time` function (the entire function body, roughly lines 100–120):

```python
def log_function_time(
    *,
    print_only: bool = False,
    debug_only: bool = False,
    include_args: bool = False,
    include_args_subset: dict[str, Any] | None = None,
) -> Callable[[_F], _F]:
    """No-op timing decorator stub."""

    def decorator(func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
```

with:

```python
def log_function_time(
    *,
    print_only: bool = False,
    debug_only: bool = False,
    include_args: bool = False,
    include_args_subset: dict[str, Any] | None = None,
) -> Callable[[_F], _F]:
    """Decorator that logs the wall-clock time of the wrapped function.

    Logs at DEBUG when debug_only=True, otherwise at INFO.
    print_only is accepted for call-site compatibility but has no effect
    (all output goes through the standard logger).
    """
    import time

    def decorator(func: _F) -> _F:
        logger = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                msg = f"{func.__qualname__} took {elapsed:.3f}s"
                if debug_only:
                    logger.debug(msg)
                else:
                    logger.info(msg)

        return wrapper  # type: ignore[return-value]

    return decorator
```

Note: `import time` is placed inside `decorator` to avoid a module-level import for a rarely-needed stdlib. If you prefer top-level imports, add `import time` at the top of `utils.py` alongside the existing imports and remove the inner `import time`.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/document_index/test_types.py::test_log_function_time_logs_timing -v
```

Expected: PASS.

- [ ] **Step 5: Run all unit tests**

```bash
pytest tests/unit/ -v -x
```

Expected: all pass.

- [ ] **Step 6: Run linter**

```bash
ruff check src/backend/document_index/utils.py --fix && ruff format src/backend/document_index/utils.py
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/backend/document_index/utils.py tests/unit/document_index/test_types.py
git commit -m "feat(utils): replace log_function_time no-op with real perf_counter timer"
```

---

## Self-Review

### Spec coverage
- ✅ `vector_db_insertion.py` compat shim deleted — Task 1
- ✅ `indexing.py` compat alias removed — Task 1
- ✅ `interfaces_new.py` renamed to `interfaces.py` — Task 2
- ✅ All 8 import sites updated — Task 2
- ✅ `log_function_time` no-op replaced with real timer — Task 3
- ✅ Tests updated to match new names — Tasks 2 and 3

### Scope note on `text_processor.py`
`src/retrieval/text_processor.py` is correctly placed: it's used by `vocabulary.py` (which builds the BM25 vocabulary) and has its own unit tests. It does not overlap with `chunk_content_enrichment.py` (which removes RAG augmentations from retrieved chunks). No changes needed.

### Scope note on old chunk ID functions
`get_uuid_from_chunk_info_old` and `get_uuid_from_chunk_old` in `document_index_utils.py` are called internally within `get_document_chunk_ids()` for migration backward-compat. They have unit tests. Leave them in place.

### Placeholder scan
No TBD, TODO, or vague steps found.

### Type consistency
`log_function_time` signature is identical before and after — all 25+ call sites in `client.py` need zero changes.
