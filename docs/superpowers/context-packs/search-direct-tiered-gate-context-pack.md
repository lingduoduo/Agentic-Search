# Generated Context Pack

# Search Direct Tiered Gate

## Sources

- [Specification: 2026-07-07-search-direct-tiered-gate-design.md](../specs/2026-07-07-search-direct-tiered-gate-design.md)
- [Plan: 2026-07-07-search-direct-tiered-gate.md](../plans/2026-07-07-search-direct-tiered-gate.md)

## Specification Context

### Goal

Replace the single score threshold with a **tiered match cascade** between the
query and the rank-1 retrieval result. Each tier is backend-independent (string
ops or a fixed-scale cosine), so the gate behaves identically on TF-IDF, RRF, or
dense. The cascade is biased toward the cheap no-LLM `direct_retrieval` path:
any tier that fires routes direct; only when none fire do we escalate to search.

### Architecture / touch points

1. **New pure helper `_direct_gate_decision(query, docs, *, cos_min, embedder)
   -> (is_strong: bool, tier: str, top_score: float, cosine: float | None)`**
   in `src/internal/servers/web/app.py`. No I/O except the injected `embedder`
   call in Tiers 2/3. Fully unit-testable with a fake `EmbeddingFn`.
2. **`_run_search_direct_or_escalate`** (`app.py:857`) — replace
   `if real and top_score >= threshold:` with
   `is_strong, tier, top_score, cosine = _direct_gate_decision(query, real, ...)`
   then `if is_strong:`. The strong-return `extra` gains `"tier"`; the escalate
   path is unchanged.
3. **Levenshtein.** Small internal helper (bounded early-exit at distance 2 — we

…

### Testing

- **Unit — `_direct_gate_decision`** with an injected deterministic
  `EmbeddingFn` (repo has `deterministic_embedding_fn` / `numpy_embedding_fn`),
  never a real model:
  - exact title match → direct, tier `exact`, embedder **not** called.
  - typo (Levenshtein 1) + high cosine → direct, tier `fuzzy`.
  - typo (Levenshtein 1) + low cosine (`cat`/`car`) → search.
  - no lexical match + cosine > 0.8 → direct, tier `semantic`.
  - no lexical match + cosine ≤ 0.8 → search, tier `weak`.
  - empty docs → search.
  - embedder `None` (model unavailable) + only-semantic-would-match → search,
    no crash.
- **Unit — backend independence:** the same query/result pair routes `exact`

…

## Implementation Plan Context

### Global Constraints

- The gate compares the query against the **rank-1** retrieval result only (`docs[0]`, already rerank/MMR-ordered).
- Tiers are OR'd, short-circuit cheapest-first; any tier firing → direct. Only when none fire → escalate to search (`_escalate`, unchanged).
- **No score-threshold fallback** — `_search_direct_min_score` and `AGENTIC_SEARCH_SEARCH_DIRECT_MIN_SCORE` are removed from the gate.
- Semantic tier uses a **fixed-scale** cosine so `AGENTIC_SEARCH_SEARCH_DIRECT_COS_MIN` (default `0.8`) means the same on every backend.
- Missing/unavailable embedding model → `cosine_fn` returns `None` → semantic + fuzzy-verify no-op → escalate. **Never crash the hot path.**

…

### Task 1: Pure string helpers — `_norm` + `_levenshtein_lt2`

**Files:**
- Modify: `src/internal/servers/web/app.py` (add helpers immediately before `def _search_direct_min_score()` ~line 764; ensure `import re` present near the top imports)
- Test: `tests/unit/test_direct_gate.py` (new)

**Interfaces:**
- Produces: `_norm(text: str) -> str`; `_levenshtein_lt2(a: str, b: str) -> bool` (True iff edit distance is 0 or 1).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_direct_gate.py`:

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_direct_gate.py -v`
Expected: FAIL — `ImportError: cannot import name '_levenshtein_lt2'`.

- [ ] **Step 3: Add the helpers**

…

### Task 2: `_direct_gate_decision` — the pure tiered cascade

**Files:**
- Modify: `src/internal/servers/web/app.py` (add after `_levenshtein_lt2`)
- Test: `tests/unit/test_direct_gate.py` (append)

**Interfaces:**
- Consumes: `_norm`, `_levenshtein_lt2` (Task 1); `ContextDocument` (`.title: str`, `.content: str`, `.score: float`, ordered rank-1 first).
- Produces: `_direct_gate_decision(query: str, docs: list[ContextDocument], *, cos_min: float, cosine_fn: Callable[[str, str], float | None]) -> tuple[bool, str, float, float | None]` returning `(is_strong, tier, top_score, cosine)` where `tier` ∈ `{"exact","fuzzy","semantic","weak"}`. `cosine_fn(query, passage)` returns the cosine or `None` when no embedder is available.

…

### Task 3: Embedding plumbing — `_gate_embedder` + `_make_cosine_fn` + `_search_direct_cos_min`

**Files:**
- Modify: `src/internal/servers/web/app.py` (add after `_direct_gate_decision`; ensure `import numpy as np` present near top imports)
- Test: `tests/unit/test_direct_gate.py` (append)

**Interfaces:**
- Produces:
  - `_search_direct_cos_min() -> float` (reads `AGENTIC_SEARCH_SEARCH_DIRECT_COS_MIN`, default `0.8`).
  - `_gate_embedder() -> EmbeddingFn | None` — lazy module-level singleton; `None` when `AGENTIC_SEARCH_SEARCH_DIRECT_SEMANTIC=0` or the model can't load. `EmbeddingFn = Callable[[list[str]], np.ndarray]`.

…

### Task 4: Wire the gate into `_run_search_direct_or_escalate`; remove the score threshold

**Files:**
- Modify: `src/internal/servers/web/app.py` (`_run_search_direct_or_escalate`: delete `threshold = _search_direct_min_score()` ~line 791; replace the strong/weak block ~lines 842–882; delete `_search_direct_min_score` ~lines 764–767)
- Modify: `tests/conftest.py` (disable the gate's model load suite-wide)
- Modify: `tests/unit/test_execution_fallbacks.py` (`_doc` + `test_strong_retrieval_returns_direct_without_agent`)

**Interfaces:**
- Consumes: `_direct_gate_decision`, `_search_direct_cos_min`, `_make_cosine_fn`, `_gate_embedder` (Tasks 2–3); existing `_run_direct_search`, `_escalate`, `_search_only_answer`, `_capture.record_stage`.

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
