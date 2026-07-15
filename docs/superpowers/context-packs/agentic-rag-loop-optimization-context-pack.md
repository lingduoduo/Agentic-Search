# Generated Context Pack

# Agentic Rag Loop Optimization

## Sources

- [Specification: 2026-07-01-agentic-rag-loop-optimization-design.md](../specs/2026-07-01-agentic-rag-loop-optimization-design.md)
- [Plan: 2026-07-01-agentic-rag-loop-optimization.md](../plans/2026-07-01-agentic-rag-loop-optimization.md)

## Specification Context

### In scope (6 changes)

| # | Change | Type |
|---|--------|------|
| 1 | Initialize `merged` before the loop | Correctness (crash guard) |
| 2 | Normalize queries for dedup bookkeeping | Robustness |
| 3 | Robust document dedup key (normalized URL / content hash) | Robustness |
| 4 | Cap follow-up queries per round (`max_followups_per_round=5`) | Cost control |
| 5 | De-duplicate the `_clean_line` call in `_parse_gap_queries` | Cleanup |
| 6 | Bounded timeout on the LLM sufficiency check | Robustness (no hang) |

### Explicitly NOT in scope (decided with user)

- **Do not** flip the sufficiency-check fail default. On judge failure/timeout the loop
  still returns `True` (fail-open, stop looping). We bound it with a timeout instead.
- **Do not** add an empty-evidence canned fallback. When all retrieval fails the loop
  still calls `generate_answer` with zero docs (current behavior preserved).
- **Do not** add a timeout to gap analysis (`_generate_followup`) — only the
  sufficiency check.
- No other refactors, renames, prompt edits, or adjacent "improvements".

## Implementation Plan Context

### Task 1: Crash guard — initialize `merged` before the loop

**Files:**
- Modify: `src/agents/agentic_rag.py` (inside `run`, just before `for round_idx in range(...)`)
- Test: `tests/unit/test_agentic_rag.py`

**Interfaces:**
- Consumes: existing `AgenticRAGLoop(config, llm)`, `run(question)`.
- Produces: `run` never raises `UnboundLocalError` for `merged` when `max_rounds == 0` or no novel queries exist.

- [ ] **Step 1: Write the failing test**

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_agentic_rag.py::test_run_with_zero_max_rounds_returns_empty_context -v`
Expected: FAIL with `UnboundLocalError: local variable 'merged' referenced before assignment`

- [ ] **Step 3: Write minimal implementation**

…

### Task 2: Normalized, intra-batch query deduplication

**Files:**
- Modify: `src/agents/agentic_rag.py` (add helpers `_norm_query`, `_dedupe_novel`; rewrite the `seen_queries` bookkeeping in `run`)
- Test: `tests/unit/test_agentic_rag.py`

**Interfaces:**
- Produces:
  - `_norm_query(q: str) -> str` — lowercase + whitespace-collapsed form.
  - `_dedupe_novel(queries: list[str], seen: set[str]) -> list[str]` — returns items whose normalized form is not yet in `seen`, recording each into `seen` as it goes (dedupes within the batch and across rounds). Returned strings are the **original** (un-normalized) queries.

- [ ] **Step 1: Write the failing test**

- [ ] **Step 2: Run test to verify it fails**

…

### Task 3: Robust document dedup key

**Files:**
- Modify: `src/agents/agentic_rag.py` (add `import hashlib`; add `_doc_key`; use it in the accumulate loop)
- Test: `tests/unit/test_agentic_rag.py`

**Interfaces:**
- Produces: `_doc_key(doc: ContextDocument) -> str` — normalized URL (`strip().lower()`) when a URL is present, else the SHA-256 hex digest of the full content.

- [ ] **Step 1: Write the failing test**

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_agentic_rag.py::test_url_less_docs_dedup_by_full_content -v`

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
