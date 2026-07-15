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

### Non-goals

- Not changing intent classification, the SearchAgentLoop, or the CHAT path.
- Not changing retrieval, escalation, the `explicit_source` bypass, the
  `try/except` guard, or `_search_only_answer` — only the one strong/weak
  decision line inside `_run_search_direct_or_escalate` changes.
- Not matching the query against a large canonical-term index (no FAISS/ANN —
  the gate is a 1-vs-1 comparison against the top result).
- Not re-embedding full document content (that duplicates the retrieval server).

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
   only care about `< 2`), no new dependency.
4. **Embedder.** Reuse the existing `EmbeddingFn` protocol
   (`src/internal/document_index/indexing.py`) and the `intfloat/e5-base-v2`
   loader already used elsewhere. A **lazy module-level singleton**, warmed once
   in the FastAPI lifespan. If `sentence-transformers`/torch is unavailable or
   the load throws, the embedder is `None`: Tier 3 is skipped and Tier 2 cannot
   semantically confirm, so both fall through to `search`. The hot path never
   crashes on a missing model (same philosophy as the existing `try/except`).

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
  regardless of `top_score` magnitude (proves the RRF no-op is gone).
- **Integration:** a clean-lookup SEARCH request through `/api/agent` returns
  `search_mode: direct`, `tier: exact`, with no agent-loop stage.
- Full suite green; ruff clean.

## Implementation Plan Context

### Global Constraints

- The gate compares the query against the **rank-1** retrieval result only (`docs[0]`, already rerank/MMR-ordered).
- Tiers are OR'd, short-circuit cheapest-first; any tier firing → direct. Only when none fire → escalate to search (`_escalate`, unchanged).
- **No score-threshold fallback** — `_search_direct_min_score` and `AGENTIC_SEARCH_SEARCH_DIRECT_MIN_SCORE` are removed from the gate.
- Semantic tier uses a **fixed-scale** cosine so `AGENTIC_SEARCH_SEARCH_DIRECT_COS_MIN` (default `0.8`) means the same on every backend.
- Missing/unavailable embedding model → `cosine_fn` returns `None` → semantic + fuzzy-verify no-op → escalate. **Never crash the hot path.**
- Web tests must not load a real model: the suite sets `AGENTIC_SEARCH_SEARCH_DIRECT_SEMANTIC=0` in `tests/conftest.py`; gate unit tests inject a fake `cosine_fn`.
- Run `ruff check <files> --fix && ruff format <files>` before each commit (pre-commit hook; if a commit aborts because the hook reformatted, `git add -A` and re-run the same commit).
- Branch: `feat/search-direct-tiered-gate` (spec already committed there).

---

### Task 1: Pure string helpers — `_norm` + `_levenshtein_lt2`

**Files:**
- Modify: `src/internal/servers/web/app.py` (add helpers immediately before `def _search_direct_min_score()` ~line 764; ensure `import re` present near the top imports)
- Test: `tests/unit/test_direct_gate.py` (new)

**Interfaces:**
- Produces: `_norm(text: str) -> str`; `_levenshtein_lt2(a: str, b: str) -> bool` (True iff edit distance is 0 or 1).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_direct_gate.py`:

```python
from src.internal.servers.web.app import _levenshtein_lt2, _norm


def test_norm_lowercases_strips_and_collapses():
    assert _norm("  FAISS? ") == "faiss"
    assert _norm("Dense   Retrieval") == "dense retrieval"


def test_levenshtein_lt2_true_for_zero_and_one_edit():
    assert _levenshtein_lt2("faiss", "faiss") is True   # distance 0
    assert _levenshtein_lt2("faiss", "faisz") is True   # 1 substitution
    assert _levenshtein_lt2("faiss", "faisss") is True  # 1 insertion
    assert _levenshtein_lt2("faisss", "faiss") is True  # 1 deletion


def test_levenshtein_lt2_false_for_two_or_more_edits():
    assert _levenshtein_lt2("cat", "dog") is False
    assert _levenshtein_lt2("faiss", "hnsw") is False
    assert _levenshtein_lt2("faiss", "fabss") is True   # exactly 1 sub → still True
    assert _levenshtein_lt2("faiss", "fabsz") is False  # 2 subs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_direct_gate.py -v`
Expected: FAIL — `ImportError: cannot import name '_levenshtein_lt2'`.

- [ ] **Step 3: Add the helpers**

_[Section compacted.]_

### Task 2: `_direct_gate_decision` — the pure tiered cascade

**Files:**
- Modify: `src/internal/servers/web/app.py` (add after `_levenshtein_lt2`)
- Test: `tests/unit/test_direct_gate.py` (append)

**Interfaces:**
- Consumes: `_norm`, `_levenshtein_lt2` (Task 1); `ContextDocument` (`.title: str`, `.content: str`, `.score: float`, ordered rank-1 first).
- Produces: `_direct_gate_decision(query: str, docs: list[ContextDocument], *, cos_min: float, cosine_fn: Callable[[str, str], float | None]) -> tuple[bool, str, float, float | None]` returning `(is_strong, tier, top_score, cosine)` where `tier` ∈ `{"exact","fuzzy","semantic","weak"}`. `cosine_fn(query, passage)` returns the cosine or `None` when no embedder is available.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_direct_gate.py`:

```python
from src.context.models import ContextDocument
from src.internal.servers.web.app import _direct_gate_decision


def _d(title, *, score=0.5, content="body", i=1):
    return ContextDocument(
        id=f"D{i}", title=title, content=content, url=None, score=score
    )


def test_exact_title_routes_direct_without_touching_embedder():
    calls = {"n": 0}

    def cos(q, p):
        calls["n"] += 1
        return 0.0

    strong, tier, top, cosine = _direct_gate_decision(
        "FAISS", [_d("faiss"), _d("other", score=0.1, i=2)], cos_min=0.8, cosine_fn=cos
    )
    assert (strong, tier) == (True, "exact")
    assert calls["n"] == 0  # exact match short-circuits before any embed
    assert cosine is None
    assert top == 0.5


def test_fuzzy_typo_with_high_cosine_routes_direct():

_[Section compacted.]_

### Task 3: Embedding plumbing — `_gate_embedder` + `_make_cosine_fn` + `_search_direct_cos_min`

**Files:**
- Modify: `src/internal/servers/web/app.py` (add after `_direct_gate_decision`; ensure `import numpy as np` present near top imports)
- Test: `tests/unit/test_direct_gate.py` (append)

**Interfaces:**
- Produces:
  - `_search_direct_cos_min() -> float` (reads `AGENTIC_SEARCH_SEARCH_DIRECT_COS_MIN`, default `0.8`).
  - `_gate_embedder() -> EmbeddingFn | None` — lazy module-level singleton; `None` when `AGENTIC_SEARCH_SEARCH_DIRECT_SEMANTIC=0` or the model can't load. `EmbeddingFn = Callable[[list[str]], np.ndarray]`.
  - `_make_cosine_fn(embedder) -> Callable[[str, str], float | None]` — applies e5 `query:`/`passage:` prefixes, L2-normalizes, returns cosine (or `None` if `embedder is None` / encode fails / zero-norm).

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_direct_gate.py`:

```python
import numpy as np

import src.internal.servers.web.app as web_app
from src.internal.servers.web.app import (
    _gate_embedder,
    _make_cosine_fn,
    _search_direct_cos_min,
)


def test_cos_min_default_and_override(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_SEARCH_DIRECT_COS_MIN", raising=False)
    assert _search_direct_cos_min() == 0.8
    monkeypatch.setenv("AGENTIC_SEARCH_SEARCH_DIRECT_COS_MIN", "0.7")
    assert _search_direct_cos_min() == 0.7


def test_make_cosine_fn_none_embedder_returns_none():
    assert _make_cosine_fn(None)("a", "b") is None


def test_make_cosine_fn_identical_vectors_cosine_one():
    emb = lambda texts: np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)  # noqa: E731

_[Section compacted.]_

### Task 4: Wire the gate into `_run_search_direct_or_escalate`; remove the score threshold

**Files:**
- Modify: `src/internal/servers/web/app.py` (`_run_search_direct_or_escalate`: delete `threshold = _search_direct_min_score()` ~line 791; replace the strong/weak block ~lines 842–882; delete `_search_direct_min_score` ~lines 764–767)
- Modify: `tests/conftest.py` (disable the gate's model load suite-wide)
- Modify: `tests/unit/test_execution_fallbacks.py` (`_doc` + `test_strong_retrieval_returns_direct_without_agent`)

**Interfaces:**
- Consumes: `_direct_gate_decision`, `_search_direct_cos_min`, `_make_cosine_fn`, `_gate_embedder` (Tasks 2–3); existing `_run_direct_search`, `_escalate`, `_search_only_answer`, `_capture.record_stage`.
- Produces: strong-path `extra` now carries `{"search_mode": "direct", "tier": tier, "top_score": top_score}`; `sufficiency` capture carries `tier` + `cosine`.

- [ ] **Step 1: Disable the gate model load across the test suite**

In `tests/conftest.py`, beside the existing `os.environ["SEARCH_AGENT_MODEL"] = ""` lines, add:

```python
os.environ["AGENTIC_SEARCH_SEARCH_DIRECT_SEMANTIC"] = "0"
```

- [ ] **Step 2: Update the existing dispatch tests for the new gate**

In `tests/unit/test_execution_fallbacks.py`, change `_doc` to accept a title and update the strong test to use an exact title match (the old test relied on the removed `0.2` score threshold). Replace the `_doc` definition and `test_strong_retrieval_returns_direct_without_agent`:

```python
def _doc(score: float, i: int = 1, title: str | None = None) -> ContextDocument:
    return ContextDocument(
        id=f"D{i}",
        title=title or f"doc{i}",

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
