# SEARCH Direct-First Tiered Match Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single `top_score >= 0.2` strong/weak gate in `_run_search_direct_or_escalate` with a backend-independent tiered cascade (exact title → fuzzy+semantic-verify → semantic → else escalate).

**Architecture:** Four small, isolated units in `src/internal/servers/web/app.py` — pure string helpers (`_norm`, `_levenshtein_lt2`), a pure decision function (`_direct_gate_decision`) that takes an injected `cosine_fn` so it is testable with zero model load, and the model plumbing (`_gate_embedder` lazy e5 singleton + `_make_cosine_fn`). The old score-threshold helper is deleted. Only one decision line inside the existing dispatch changes; retrieval, escalation, and capture stay put.

**Tech Stack:** Python 3.12, FastAPI, numpy, sentence-transformers (`intfloat/e5-base-v2`, reused, lazy), pytest.

## Global Constraints

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

In `src/internal/servers/web/app.py`, confirm `import re` exists near the top (add it beside the other stdlib imports if missing). Then add immediately before `def _search_direct_min_score()`:

```python
_GATE_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    """Lowercase, trim surrounding punctuation/space, collapse inner whitespace."""
    text = (text or "").strip().lower().strip(".,!?;:\"'()[]{}").strip()
    return _GATE_WS.sub(" ", text)


def _levenshtein_lt2(a: str, b: str) -> bool:
    """True iff the Levenshtein distance between a and b is 0 or 1 (bounded, O(n))."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(1 for x, y in zip(a, b) if x != y) <= 1
    if la > lb:
        a, b = b, a  # ensure a is the shorter string
    i = j = 0
    skipped = False
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            if skipped:
                return False
            skipped = True
            j += 1  # consume one extra char from the longer string
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_direct_gate.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
ruff check src/internal/servers/web/app.py tests/unit/test_direct_gate.py --fix && ruff format src/internal/servers/web/app.py tests/unit/test_direct_gate.py
git add src/internal/servers/web/app.py tests/unit/test_direct_gate.py
git commit -m "feat(search): add _norm and bounded _levenshtein_lt2 gate helpers"
```

---

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
    strong, tier, _top, cosine = _direct_gate_decision(
        "faisz", [_d("faiss")], cos_min=0.8, cosine_fn=lambda q, p: 0.95
    )
    assert (strong, tier) == (True, "fuzzy")
    assert cosine == 0.95


def test_fuzzy_near_match_with_low_cosine_escalates():
    # "car" vs "cat": edit distance 1 but unrelated → semantic verify fails.
    strong, tier, _top, _cos = _direct_gate_decision(
        "car", [_d("cat")], cos_min=0.8, cosine_fn=lambda q, p: 0.1
    )
    assert (strong, tier) == (False, "weak")


def test_semantic_equivalent_phrasing_routes_direct():
    strong, tier, _top, cosine = _direct_gate_decision(
        "vector index library", [_d("faiss")], cos_min=0.8, cosine_fn=lambda q, p: 0.9
    )
    assert (strong, tier) == (True, "semantic")
    assert cosine == 0.9


def test_semantic_low_cosine_escalates():
    strong, tier, _top, _cos = _direct_gate_decision(
        "weather today", [_d("faiss")], cos_min=0.8, cosine_fn=lambda q, p: 0.2
    )
    assert (strong, tier) == (False, "weak")


def test_no_embedder_semantic_path_escalates():
    strong, tier, _top, cosine = _direct_gate_decision(
        "vector index", [_d("faiss")], cos_min=0.8, cosine_fn=lambda q, p: None
    )
    assert (strong, tier) == (False, "weak")
    assert cosine is None


def test_empty_docs_escalate():
    strong, tier, top, cosine = _direct_gate_decision(
        "faiss", [], cos_min=0.8, cosine_fn=lambda q, p: 0.9
    )
    assert (strong, tier, top, cosine) == (False, "weak", 0.0, None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_direct_gate.py -k "exact or fuzzy or semantic or empty_docs or no_embedder" -v`
Expected: FAIL — `ImportError: cannot import name '_direct_gate_decision'`.

- [ ] **Step 3: Add the decision function**

In `src/internal/servers/web/app.py`, add after `_levenshtein_lt2` (ensure `from collections.abc import Callable` or `from typing import Callable` is available — the file already imports typing helpers; add `Callable` to the existing import if absent):

```python
def _direct_gate_decision(
    query: str,
    docs: list[ContextDocument],
    *,
    cos_min: float,
    cosine_fn: "Callable[[str, str], float | None]",
) -> tuple[bool, str, float, float | None]:
    """Tiered strong/weak gate over the rank-1 retrieval result.

    exact title match → direct; typo (Levenshtein<2) confirmed by cosine → direct;
    semantic cosine > cos_min → direct; otherwise escalate. Backend-independent.
    """
    top_score = max((d.score or 0.0 for d in docs), default=0.0)
    if not docs:
        return False, "weak", top_score, None

    top = docs[0]
    q = _norm(query)
    t = _norm(top.title)

    if q and q == t:
        return True, "exact", top_score, None

    passage = f"{top.title} {top.content}".strip()

    if q and t and _levenshtein_lt2(q, t):
        cosine = cosine_fn(query, passage)
        if cosine is not None and cosine > cos_min:
            return True, "fuzzy", top_score, cosine
        return False, "weak", top_score, cosine

    cosine = cosine_fn(query, passage)
    if cosine is not None and cosine > cos_min:
        return True, "semantic", top_score, cosine
    return False, "weak", top_score, cosine
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_direct_gate.py -v`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
ruff check src/internal/servers/web/app.py tests/unit/test_direct_gate.py --fix && ruff format src/internal/servers/web/app.py tests/unit/test_direct_gate.py
git add src/internal/servers/web/app.py tests/unit/test_direct_gate.py
git commit -m "feat(search): tiered direct-gate decision (exact/fuzzy/semantic)"
```

---

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
    assert abs(_make_cosine_fn(emb)("x", "y") - 1.0) < 1e-6


def test_make_cosine_fn_orthogonal_vectors_cosine_zero():
    emb = lambda texts: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # noqa: E731
    assert abs(_make_cosine_fn(emb)("x", "y")) < 1e-6


def test_make_cosine_fn_encode_failure_returns_none():
    def emb(texts):
        raise RuntimeError("boom")

    assert _make_cosine_fn(emb)("x", "y") is None


def test_gate_embedder_disabled_by_env_returns_none(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_SEARCH_DIRECT_SEMANTIC", "0")
    web_app._GATE_EMBEDDER = None  # reset singleton cache
    assert _gate_embedder() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_direct_gate.py -k "cos_min or make_cosine or gate_embedder" -v`
Expected: FAIL — `ImportError: cannot import name '_gate_embedder'`.

- [ ] **Step 3: Add the plumbing**

In `src/internal/servers/web/app.py`, confirm `import numpy as np` is present near the top imports (add if missing). Add after `_direct_gate_decision`:

```python
_GATE_EMBEDDER: object | None = None  # None=unset, False=failed, callable=loaded


def _search_direct_cos_min() -> float:
    import os as _os

    return float(_os.environ.get("AGENTIC_SEARCH_SEARCH_DIRECT_COS_MIN", "0.8"))


def _gate_embedder():
    """Lazy singleton e5 embedder for the semantic tier; None when unavailable."""
    global _GATE_EMBEDDER
    import os as _os

    if _os.environ.get("AGENTIC_SEARCH_SEARCH_DIRECT_SEMANTIC", "1") == "0":
        return None
    if _GATE_EMBEDDER is not None:
        return _GATE_EMBEDDER or None
    try:
        from sentence_transformers import SentenceTransformer

        name = _os.environ.get(
            "AGENTIC_SEARCH_SEARCH_DIRECT_MODEL", "intfloat/e5-base-v2"
        )
        model = SentenceTransformer(name)

        def _fn(texts):
            return model.encode(texts, normalize_embeddings=True)

        _GATE_EMBEDDER = _fn
    except Exception:
        logger.exception(
            "direct-gate: embedding model unavailable — semantic tier disabled"
        )
        _GATE_EMBEDDER = False
        return None
    return _GATE_EMBEDDER


def _make_cosine_fn(embedder):
    """Return (query, passage) -> cosine|None using e5 prefixes; None if no model."""
    if embedder is None:
        return lambda _query, _passage: None

    def _cosine(query: str, passage: str):
        try:
            vecs = embedder([f"query: {query}", f"passage: {passage}"])
        except Exception:
            return None
        qv = np.asarray(vecs[0], dtype=np.float32)
        pv = np.asarray(vecs[1], dtype=np.float32)
        qn = float(np.linalg.norm(qv))
        pn = float(np.linalg.norm(pv))
        if qn == 0.0 or pn == 0.0:
            return None
        return float(np.dot(qv, pv) / (qn * pn))

    return _cosine
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_direct_gate.py -v`
Expected: PASS (all Task 1–3 tests).

- [ ] **Step 5: Commit**

```bash
ruff check src/internal/servers/web/app.py tests/unit/test_direct_gate.py --fix && ruff format src/internal/servers/web/app.py tests/unit/test_direct_gate.py
git add src/internal/servers/web/app.py tests/unit/test_direct_gate.py
git commit -m "feat(search): lazy e5 embedder + cosine fn for direct-gate semantic tier"
```

---

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
        content="body",
        url=None,
        score=score,
        metadata={},
    )
```

```python
def test_strong_retrieval_returns_direct_without_agent(monkeypatch):
    # Exact title match ("FAISS" == title) → direct, agent loop NOT called.
    (answer, citations, documents, intent, extra), called = _call_direct_or_escalate(
        monkeypatch, [_doc(0.42, title="FAISS"), _doc(0.1, 2)]
    )
    assert called["agent"] is False
    assert extra["search_mode"] == "direct"
    assert extra["tier"] == "exact"
    assert documents[0].score == 0.42
    assert intent == "search"
```

(`_call_direct_or_escalate` already passes query `"FAISS"`; `test_weak_retrieval_escalates_to_agent` and `test_empty_retrieval_escalates` are unchanged — their `doc1`/empty results miss every tier and escalate.)

- [ ] **Step 3: Run the updated tests to verify they fail**

Run: `python -m pytest tests/unit/test_execution_fallbacks.py -k "strong_retrieval or weak_retrieval or empty_retrieval" -v`
Expected: FAIL — `test_strong_retrieval...` fails on `KeyError: 'tier'` (the helper doesn't emit `tier` yet).

- [ ] **Step 4: Delete `_search_direct_min_score` and wire the gate**

In `src/internal/servers/web/app.py`:

(a) Delete the helper (~lines 764–767):

```python
def _search_direct_min_score() -> float:
    import os as _os

    return float(_os.environ.get("AGENTIC_SEARCH_SEARCH_DIRECT_MIN_SCORE", "0.2"))
```

(b) In `_run_search_direct_or_escalate`, delete the line `threshold = _search_direct_min_score()`.

(c) Replace the block that starts at `real = [d for d in documents if not d.metadata.get("error")]` and runs through `return await _escalate(top_score, "weak_retrieval")` with:

```python
    real = [d for d in documents if not d.metadata.get("error")]
    is_strong, tier, top_score, cosine = _direct_gate_decision(
        query,
        real,
        cos_min=_search_direct_cos_min(),
        cosine_fn=_make_cosine_fn(_gate_embedder()),
    )
    _capture.record_stage(
        "search",
        "direct_retrieval",
        {
            "query": query,
            "top_k": top_k,
            "top_score": top_score,
            "documents": [
                {"id": d.id, "title": d.title, "score": d.score} for d in real
            ],
        },
    )

    if is_strong:
        _capture.record_stage(
            "search",
            "sufficiency",
            {"mode": "direct", "tier": tier, "cosine": cosine, "top_score": top_score},
        )
        answer = _search_only_answer(
            "Direct retrieval",
            queries=[query],
            documents=real,
            source_provider="retrieval",
        )
        return (
            answer,
            [d.citation for d in real],
            real,
            "search",
            {"search_mode": "direct", "tier": tier, "top_score": top_score},
        )

    _capture.record_stage(
        "search",
        "sufficiency",
        {"mode": "escalated", "tier": tier, "cosine": cosine, "top_score": top_score},
    )
    return await _escalate(top_score, "weak_retrieval")
```

- [ ] **Step 5: Run the updated + gate tests to verify they pass**

Run: `python -m pytest tests/unit/test_execution_fallbacks.py tests/unit/test_direct_gate.py -v`
Expected: PASS (strong→direct/exact, weak/empty→escalate, all gate units).

- [ ] **Step 6: Run the web suite + search tests for regressions**

Run: `python -m pytest tests/unit/servers/web/ tests/unit/test_execution_fallbacks.py tests/unit/test_direct_gate.py tests/unit/test_search_tools.py -q`
Expected: PASS — no regressions. Confirm no test loads a real model (suite finishes in seconds).

- [ ] **Step 7: Verify no dangling references to the removed helper**

Run: `grep -rn "_search_direct_min_score\|SEARCH_DIRECT_MIN_SCORE" src tests`
Expected: no output (env var is now dead; the earlier spec that documented it is historical and not edited here).

- [ ] **Step 8: Commit**

```bash
ruff check src/internal/servers/web/app.py tests/conftest.py tests/unit/test_execution_fallbacks.py --fix && ruff format src/internal/servers/web/app.py tests/conftest.py tests/unit/test_execution_fallbacks.py
git add src/internal/servers/web/app.py tests/conftest.py tests/unit/test_execution_fallbacks.py
git commit -m "feat(search): route direct-first via tiered gate, drop score threshold"
```

---

## Self-Review

**Spec coverage:**
- Tiered gate (exact/fuzzy/semantic/else) → Task 2 `_direct_gate_decision`.
- Exact = normalized title equality → Task 2 + `_norm` (Task 1).
- Fuzzy = Levenshtein<2 AND semantic-confirm → Task 2 fuzzy branch + `_levenshtein_lt2` (Task 1).
- Semantic = e5 cosine > 0.8, fixed scale, title+snippet embed → Tasks 2/3 (`passage = title + content`, `_make_cosine_fn` prefixes/normalize).
- No score fallback; `top_score>=0.2` removed → Task 4 (delete `_search_direct_min_score`, delete `threshold`).
- Reuse e5 + `EmbeddingFn`, lazy singleton, graceful None → Task 3 `_gate_embedder`.
- `COS_MIN` config default 0.8 → Task 3 `_search_direct_cos_min`.
- Observability tier + cosine on `sufficiency` → Task 4 capture stages.
- Testing (all tier branches, backend-independence via exact match regardless of score, empty docs, no-embedder, integration through the helper) → Tasks 2 + 4. Backend-independence is proven by `test_exact_title_routes_direct_without_touching_embedder` (routes on match, ignores score) and the Task-4 strong test (`score=0.42` irrelevant; routes on exact title).
- Never crash on missing model → Task 3 (`_gate_embedder` try/except → False; `_make_cosine_fn` None + encode-failure guards) + `test_make_cosine_fn_encode_failure_returns_none`, `test_no_embedder_semantic_path_escalates`.

**Placeholder scan:** every step has concrete code, exact paths, exact commands, expected output. No TBD/TODO.

**Type consistency:** `_direct_gate_decision(...) -> (bool, str, float, float|None)` is unpacked as `is_strong, tier, top_score, cosine` in Task 4. `cosine_fn: Callable[[str,str], float|None]` is produced by `_make_cosine_fn` (Task 3) and stubbed in Task 2 tests with matching `lambda q, p: ...`. `_gate_embedder() -> EmbeddingFn|None` feeds `_make_cosine_fn(embedder)` (Task 4). `ContextDocument` fields (`id/title/content/url/score/metadata`) match `src/context/models.py`. The strong-path tuple `(answer, citations, documents, "search", extra)` matches what `_run_auto_routed` already unpacks (unchanged call site).

**Note (full `/api/agent` e2e):** a live end-to-end SEARCH request needs a running retrieval server and would risk the web-test model-load hang; the direct/exact path is verified at the helper level (Task 4 strong test) instead, consistent with the existing `examples/run_web_integration_tests.sh` split.
