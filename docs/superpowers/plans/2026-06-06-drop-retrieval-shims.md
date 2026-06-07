# Drop Dead Shim Files from `src/retrieval/` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the four zero-caller compatibility shim files from `src/retrieval/` that re-export names already accessible through canonical source modules.

**Architecture:** `src/retrieval/` has four thin shims — `chunker.py`, `chunk_batch_store.py`, `indexing_pipeline.py`, `embedder.py` — that were created as bridge layers but are never imported by any code in the repo. All names they re-export are already accessible either directly from `src.retrieval.index_builder`, `src.backend.document_index.indexing`, or through the lazy `__getattr__` in `src/retrieval/__init__.py`. Deleting them is a pure subtraction with no callers to update.

**Tech Stack:** Python 3.11, pytest, ruff, git

---

## Shim File Inventory

| File | Lines | What it re-exports | Canonical source |
|------|-------|-------------------|-----------------|
| `src/retrieval/chunker.py` | 17 | `Chunker`, `chunk_document`, `chunk_documents`, `filter_indexable_documents`, `generate_large_chunks` | `src.backend.document_index.indexing` + `src.retrieval.index_builder` |
| `src/retrieval/chunk_batch_store.py` | 7 | `ChunkBatchStore` | `src.backend.document_index.indexing` |
| `src/retrieval/indexing_pipeline.py` | 21 | `DocumentBatchPrepareContext`, `embed_and_stream`, `filter_documents`, `index_document_batch`, `run_indexing_pipeline` | `src.backend.document_index.indexing` + `src.retrieval.index_builder` |
| `src/retrieval/embedder.py` | 21 | `DefaultIndexingEmbedder`, `IndexingEmbedder`, `numpy_embedding_fn`, `embed_chunks`, `embed_chunks_with_failure_handling` | `src.backend.document_index.indexing` + `src.retrieval.index_builder` |

**`src/retrieval/__init__.py` already covers all these names** — either via eager imports from `index_builder.py` or via lazy `__getattr__` redirecting to `src.backend.document_index.indexing`. No callers will break.

---

## Task 1: Verify and delete the four shim files

**Files:**
- Delete: `src/retrieval/chunker.py`
- Delete: `src/retrieval/chunk_batch_store.py`
- Delete: `src/retrieval/indexing_pipeline.py`
- Delete: `src/retrieval/embedder.py`
- Test: `tests/unit/` (run full suite to verify nothing broke)

- [ ] **Step 1: Confirm zero callers**

```bash
grep -rn \
  "from src.retrieval.chunker\|from src.retrieval.chunk_batch_store\|from src.retrieval.indexing_pipeline\|from src.retrieval.embedder\|import src.retrieval.chunker\|import src.retrieval.chunk_batch_store\|import src.retrieval.indexing_pipeline\|import src.retrieval.embedder" \
  --include="*.py" | grep -v "__pycache__"
```

Expected: **no output**. If any lines appear, update those callers to import from the canonical source (`src.retrieval.index_builder` or `src.backend.document_index.indexing`) before continuing.

- [ ] **Step 2: Delete the four files**

```bash
git rm src/retrieval/chunker.py \
       src/retrieval/chunk_batch_store.py \
       src/retrieval/indexing_pipeline.py \
       src/retrieval/embedder.py
```

Expected output:
```
rm 'src/retrieval/chunker.py'
rm 'src/retrieval/chunk_batch_store.py'
rm 'src/retrieval/indexing_pipeline.py'
rm 'src/retrieval/embedder.py'
```

- [ ] **Step 3: Run linter**

```bash
cd /Users/linghuang/Git/Agentic-Search && ruff check src/retrieval/ --fix && ruff format src/retrieval/
```

Expected: no errors.

- [ ] **Step 4: Run unit tests**

```bash
cd /Users/linghuang/Git/Agentic-Search && pytest tests/unit/ -v -x -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/linghuang/Git/Agentic-Search && git commit -m "chore(retrieval): drop four zero-caller compat shims (chunker, chunk_batch_store, indexing_pipeline, embedder)"
```

## Self-Review

### Spec coverage
- ✅ All four shims deleted — Task 1

### Placeholder scan
No TBD or vague steps found.

### Type consistency
No new types introduced.

### Out-of-scope note
`src/retrieval/__init__.py` is **not modified** — its lazy `__getattr__` for `Chunker`, `ChunkBatchStore`, etc. still works correctly because it redirects to the canonical `src.backend.document_index.indexing`, not to the shim files.
