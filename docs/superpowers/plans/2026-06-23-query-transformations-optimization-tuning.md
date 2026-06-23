# Query Transformations — Optimization & Tuning (M9) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tune the existing M1–M8 query-transform stack for higher retrieval quality and lower cost, adding a canonical query-rewrite leg inside existing modules — no new files.

**Architecture:** Each technique already lives behind `QueryTransformPipeline` (leaf) wrapped by async/cache/routed layers. We (a) add a `rewrite` leg as a new config flag + `QueryEnhancer.rewrite()` method threaded through the existing bundle/pipeline, (b) tighten transform prompts, (c) refine fusion's adaptive MMR λ, and (d) make `QueryRouter` skip expensive legs on easy queries — which is also the primary latency/cost lever. Smarter routing > running every leg blindly.

**Tech Stack:** Python 3, dataclasses, scikit-learn (joblib router model), pytest, ruff.

## Global Constraints

- Never commit to `main`. Branch `feat/query-transform-optimization-tuning` is already created.
- All `QT_*` env flags MUST keep defaulting to disabled (zero overhead when unset).
- No new files/modules; edits stay inside the listed pre-existing files.
- No change to existing public signatures `transform(...)` / `retrieval_variants(...)`; the only new public surfaces are the `rewrite` config flag/field/method and the `"rewrite"` router label (pre-approved).
- Fallback-safe: any LLM failure in a leg degrades that leg to empty/None; pipeline continues.
- Acceptance gates (all three): `eval_runner` recall@10/nDCG@10 ≥ `data/eval/baseline_metrics.json` with a measurable gain on one metric and p99 SLO held; `query_transform_benchmark` grid justifies tuned defaults; full `pytest` green with no regressions.
- `ruff check . --fix && ruff format .` clean before each commit.

---

### Task 1: Canonical query-rewrite leg (end-to-end)

Adds a single normalized/canonical rewrite of the query (typo + verbosity cleanup), distinct from step-back's broadening. Wired through config → enhancer → pipeline → bundle → variant ordering → cache signature.

**Files:**
- Modify: `src/context/query_transform.py` (config field, signature, bundle field + ordering, `_build_jobs`, `_assemble`, `from_env`)
- Modify: `src/context/query_enhancer.py` (add `_REWRITE_PROMPT` + `QueryEnhancer.rewrite`)
- Test: `tests/unit/test_query_enhancer.py`, `tests/unit/test_query_transform.py`

**Interfaces:**
- Produces: `QueryEnhancer.rewrite(query: str) -> str | None`; `QueryTransformConfig.rewrite: bool`; `TransformedQueryBundle.rewrite: str | None`; env flag `QT_REWRITE`.
- Consumes: existing `_llm_text`, `ChatMessage`, `QueryEnhancer` fallback pattern.

- [ ] **Step 1: Write the failing enhancer test**

In `tests/unit/test_query_enhancer.py`:

```python
def test_rewrite_returns_cleaned_query():
    llm = _llm("What is FAISS?")
    enhancer = QueryEnhancer(llm)
    result = enhancer.rewrite("uhh wht is  faiss???")
    assert result == "What is FAISS?"


def test_rewrite_falls_back_to_none_without_llm():
    assert QueryEnhancer(None).rewrite("anything") is None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_query_enhancer.py::test_rewrite_returns_cleaned_query -v`
Expected: FAIL — `AttributeError: 'QueryEnhancer' object has no attribute 'rewrite'`

- [ ] **Step 3: Implement `rewrite` in `query_enhancer.py`**

Add the prompt constant after `_STEP_BACK_PROMPT` (line ~33):

```python
_REWRITE_PROMPT = """Rewrite the following search query into one clear, canonical question.
Fix typos and remove filler, but preserve the original meaning and all key terms.
Return only the rewritten query, no explanation.

Query: {query}
Rewritten query:""".strip()
```

Add the method to `QueryEnhancer` (after `step_back`, before `enhance`):

```python
    def rewrite(self, query: str) -> str | None:
        """Return a cleaned, canonical rewrite of the query. None on failure/no LLM."""
        if self.llm is None:
            return None
        try:
            raw = _llm_text(
                self.llm.complete(
                    [ChatMessage(role="user", content=_REWRITE_PROMPT.format(query=query))]
                )
            ).strip()
            return _clean_line(raw) or None
        except Exception as exc:
            logger.warning("Query rewrite failed: %s", exc)
            return None
```

- [ ] **Step 4: Run to verify enhancer tests pass**

Run: `pytest tests/unit/test_query_enhancer.py -k rewrite -v`
Expected: PASS (both tests)

- [ ] **Step 5: Write the failing pipeline/bundle test**

In `tests/unit/test_query_transform.py`:

```python
def test_rewrite_flag_threads_into_bundle_and_variants():
    from unittest.mock import MagicMock
    from src.context.query_transform import QueryTransformConfig, QueryTransformPipeline

    llm = MagicMock()
    llm.complete.return_value = "what is faiss"  # rewrite output
    cfg = QueryTransformConfig(rewrite=True, max_variants=5)
    bundle = QueryTransformPipeline(cfg, llm).transform("uhh wht is faiss??")
    assert bundle.rewrite == "what is faiss"
    assert "what is faiss" in bundle.retrieval_variants(max_variants=5)
    assert bundle.retrieval_variants()[-1] == "uhh wht is faiss??"  # original always last


def test_rewrite_in_config_signature():
    from src.context.query_transform import QueryTransformConfig, config_signature

    on = config_signature(QueryTransformConfig(rewrite=True))
    off = config_signature(QueryTransformConfig(rewrite=False))
    assert on != off
```

- [ ] **Step 6: Run to verify failure**

Run: `pytest tests/unit/test_query_transform.py -k rewrite -v`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'rewrite'`

- [ ] **Step 7: Wire `rewrite` through `query_transform.py`**

In `QueryTransformConfig` add field (after `multi_query`, before `max_variants`):

```python
    multi_query: bool = False
    rewrite: bool = False
    max_variants: int = 5
```

In `config_signature`, add before the `mv=` entry:

```python
            f"r={int(getattr(config, 'rewrite', False))}",
```

In `TransformedQueryBundle` add field (after `multi_query`):

```python
    multi_query: list[str] = field(default_factory=list)
    rewrite: str | None = None
```

In `retrieval_variants`, add the rewrite candidate immediately after the `multi_query` loop (before `_add(self.hyde_text)`):

```python
        for q in self.multi_query:
            _add(q)
        _add(self.rewrite)
        _add(self.hyde_text)
```

In `_build_jobs`, add after the `multi_query` block:

```python
        if config.rewrite:
            jobs["rewrite"] = lambda: self._enhancer.rewrite(query)
```

In `_assemble`, add to the `TransformedQueryBundle(...)` constructor:

```python
            multi_query=results.get("multi_query") or [],
            rewrite=results.get("rewrite"),
        )
```

In `from_env`, add to the config build and the `any([...])` enable check:

```python
            multi_query=_bool("QT_MULTI_QUERY"),
            rewrite=_bool("QT_REWRITE"),
            max_variants=_parse_max_variants(),
```
```python
                config.multi_query,
                config.rewrite,
            ]
```

- [ ] **Step 8: Run pipeline tests + full transform module**

Run: `pytest tests/unit/test_query_transform.py tests/unit/test_query_enhancer.py -v`
Expected: PASS (new + all pre-existing)

- [ ] **Step 9: Lint + commit**

```bash
ruff check . --fix && ruff format .
git add src/context/query_transform.py src/context/query_enhancer.py \
        tests/unit/test_query_transform.py tests/unit/test_query_enhancer.py
git commit -m "feat(query-transform): add canonical query-rewrite leg (QT_REWRITE)"
```

---

### Task 2: Tighten transform prompts for recall

Quality lever validated by the eval gate (Task 5), not by asserting exact LLM text. Keep changes surgical; verify existing fallback tests stay green.

**Files:**
- Modify: `src/context/query_enhancer.py` (`_DECOMPOSE_PROMPT`, `_HYDE_PROMPT`, `_STEP_BACK_PROMPT`)
- Modify: `src/internal/retrieval/multi_query.py` (paraphrase prompt)
- Test: `tests/unit/test_query_enhancer.py`, `tests/unit/retrieval/test_multi_query.py` (existing — must stay green)

- [ ] **Step 1: Run the existing prompt tests as a baseline**

Run: `pytest tests/unit/test_query_enhancer.py tests/unit/retrieval/test_multi_query.py -v`
Expected: PASS (record count)

- [ ] **Step 2: Tighten `_DECOMPOSE_PROMPT`**

Replace the body to bias toward retrieval-friendly, non-overlapping sub-questions:

```python
_DECOMPOSE_PROMPT = """Break the question into 2-4 focused, non-overlapping sub-questions that together cover its full intent.
Each sub-question must be self-contained and keyword-rich so it retrieves well on its own.
Return one sub-question per line. Do not number them. Do not repeat the original question.
If the question is already atomic, return it unchanged on a single line.

Question: {query}""".strip()
```

- [ ] **Step 3: Tighten `_STEP_BACK_PROMPT`**

```python
_STEP_BACK_PROMPT = """Write one broader background question whose answer provides the concepts needed to answer the original.
Keep the same domain and key entities; generalise only the specifics.
Return only the rewritten question, no explanation.

Question: {query}
Broader question:""".strip()
```

- [ ] **Step 4: (multi_query) open the file and tighten the paraphrase prompt**

In `src/internal/retrieval/multi_query.py`, adjust the generation prompt so paraphrases are lexically diverse but intent-preserving (locate the existing prompt constant; add the diversity instruction):

```python
# In the prompt: add a line such as —
# "Each paraphrase must use different wording from the others; do not restate the query verbatim."
```

- [ ] **Step 5: Re-run prompt tests (fallbacks + dedup unaffected)**

Run: `pytest tests/unit/test_query_enhancer.py tests/unit/retrieval/test_multi_query.py -v`
Expected: PASS (same count as Step 1)

- [ ] **Step 6: Lint + commit**

```bash
ruff check . --fix && ruff format .
git add src/context/query_enhancer.py src/internal/retrieval/multi_query.py
git commit -m "feat(query-transform): tighten decompose/step-back/multi-query prompts for recall"
```

---

### Task 3: Refine adaptive MMR λ + fix docstring/code mismatch

`adaptive_mmr_lambda` claims "≤3 → 0.8" but checks `n <= 2`. Fix the mismatch and add a graded mid-range so medium queries lean slightly toward relevance.

**Files:**
- Modify: `src/internal/retrieval/fusion_learner.py` (`adaptive_mmr_lambda`)
- Test: `tests/unit/retrieval/test_fusion_learner.py`

**Interfaces:**
- Produces: `adaptive_mmr_lambda(query: str) -> float` with thresholds: ≤3 tokens → 0.8; 4–6 → 0.6; 7–9 → 0.5; ≥10 → 0.3.

- [ ] **Step 1: Write the failing test**

In `tests/unit/retrieval/test_fusion_learner.py`:

```python
def test_adaptive_mmr_lambda_tiers():
    from src.internal.retrieval.fusion_learner import adaptive_mmr_lambda

    assert adaptive_mmr_lambda("faiss") == 0.8            # 1 token (≤3)
    assert adaptive_mmr_lambda("a b c") == 0.8            # 3 tokens (≤3, was 0.5 before)
    assert adaptive_mmr_lambda("one two three four") == 0.6   # 4-6
    assert adaptive_mmr_lambda("one two three four five six seven") == 0.5  # 7-9
    assert adaptive_mmr_lambda(" ".join(["w"] * 12)) == 0.3  # ≥10
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/retrieval/test_fusion_learner.py::test_adaptive_mmr_lambda_tiers -v`
Expected: FAIL — `assert 0.5 == 0.8` for the 3-token case

- [ ] **Step 3: Update `adaptive_mmr_lambda`**

```python
def adaptive_mmr_lambda(query: str) -> float:
    """Return MMR lambda based on query length.

    Short  (≤3 tokens)  → 0.8  (prioritise relevance).
    Medium (4-6 tokens) → 0.6.
    Medium (7-9 tokens) → 0.5  (balanced).
    Long   (≥10 tokens) → 0.3  (prioritise diversity).
    """
    n = len(query.split())
    if n <= 3:
        return 0.8
    if n <= 6:
        return 0.6
    if n <= 9:
        return 0.5
    return 0.3
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/retrieval/test_fusion_learner.py -v`
Expected: PASS (new + existing)

- [ ] **Step 5: Lint + commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/retrieval/fusion_learner.py tests/unit/retrieval/test_fusion_learner.py
git commit -m "fix(fusion): graded adaptive MMR lambda; align ≤3-token tier with docstring"
```

---

### Task 4: Router — add `rewrite` label, improve heuristic, expand & retrain seed

Makes routing the cost lever: easy/keyword queries skip expensive LLM legs; noisy/long queries get `rewrite`. Adds the `rewrite` label end-to-end and retrains the joblib model.

**Files:**
- Modify: `src/internal/retrieval/query_router.py` (`ROUTER_LABELS`, `_heuristic`)
- Modify: `src/training/train_query_router.py` (`SEED_DATA` widened to 7 columns + comment)
- Test: `tests/unit/retrieval/test_query_router.py`

**Interfaces:**
- Consumes: `QueryTransformConfig.rewrite` (Task 1).
- Produces: `ROUTER_LABELS` with `"rewrite"` appended (index 6); `_heuristic` returns `rewrite=True` for long noisy queries.

- [ ] **Step 1: Write failing router tests**

In `tests/unit/retrieval/test_query_router.py`:

```python
def test_router_labels_include_rewrite_last():
    from src.internal.retrieval.query_router import ROUTER_LABELS
    assert ROUTER_LABELS[-1] == "rewrite"
    assert len(ROUTER_LABELS) == 7


def test_heuristic_routes_rewrite_for_long_noisy_query():
    from src.internal.retrieval.query_router import QueryRouter
    cfg = QueryRouter().predict(
        "uhh so basically what is the deal with faiss vs scann and which one is faster i think"
    )
    assert cfg.rewrite is True


def test_heuristic_short_keyword_skips_expensive_legs():
    from src.internal.retrieval.query_router import QueryRouter
    cfg = QueryRouter().predict("faiss index")
    assert cfg.hyde is False and cfg.decompose is False and cfg.multi_query is False
    assert cfg.keywords is True
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/retrieval/test_query_router.py -k "rewrite or skips" -v`
Expected: FAIL — `ROUTER_LABELS[-1]` is `"multi_query"`, and `cfg.rewrite` raises/False mismatch

- [ ] **Step 3: Add `rewrite` to `ROUTER_LABELS` and `_heuristic`**

Append to `ROUTER_LABELS`:

```python
ROUTER_LABELS = [
    "decompose",
    "hyde",
    "step_back",
    "keywords",
    "construct_filters",
    "multi_query",
    "rewrite",
]
```

Add a noise/length signal and the `rewrite` flag in `_heuristic` (extend, do not rewrite the function):

```python
_NOISE_WORDS = ("uhh", "umm", "basically", "like", "kinda", "i think", "so ")


def _heuristic(query: str) -> QueryTransformConfig:
    q = query.lower()
    tokens = query.split()
    n = len(tokens)
    has_question = any(w in q for w in _QUESTION_WORDS)
    multi_clause = (" and " in q) or (";" in q) or (", " in q) or n > 18
    has_date = bool(_DATE_RE.search(q)) or any(w in q for w in _RANGE_WORDS)
    short_keyword = n <= 3
    noisy = any(w in q for w in _NOISE_WORDS) or "??" in query
    return QueryTransformConfig(
        decompose=multi_clause,
        hyde=has_question and not short_keyword,
        step_back=has_question and not multi_clause,
        keywords=short_keyword,
        construct_filters=has_date,
        multi_query=not short_keyword and not multi_clause,
        rewrite=(noisy or n > 12) and not short_keyword,
    )
```

- [ ] **Step 4: Run router tests**

Run: `pytest tests/unit/retrieval/test_query_router.py -v`
Expected: PASS (new + existing). The learned-model path tolerates a 7th label via the existing try/except → heuristic fallback when an old 6-output model is loaded.

- [ ] **Step 5: Widen `SEED_DATA` to 7 columns and retrain**

In `src/training/train_query_router.py`, update the comment to list 7 labels and add the 7th column (rewrite) to every row, plus two noisy rewrite-positive rows. Replace `SEED_DATA`:

```python
# Labels order: decompose, hyde, step_back, keywords, construct_filters, multi_query, rewrite
SEED_DATA: list[tuple[str, list[int]]] = [
    ("faiss index", [0, 0, 0, 1, 0, 0, 0]),
    ("bm25 tuning", [0, 0, 0, 1, 0, 0, 0]),
    ("what is reciprocal rank fusion", [0, 1, 1, 0, 0, 1, 0]),
    ("how does HNSW graph search work", [0, 1, 1, 0, 0, 1, 0]),
    ("compare dense and sparse retrieval and when each wins", [1, 0, 0, 0, 0, 0, 0]),
    ("explain reranking and decompose the tradeoffs and latency", [1, 0, 0, 0, 0, 0, 0]),
    ("FAISS papers after 2023", [0, 0, 0, 0, 1, 0, 0]),
    ("arxiv papers between 2020 and 2022 on retrieval", [0, 0, 0, 0, 1, 0, 0]),
    ("best embedding model for semantic search", [0, 1, 1, 0, 0, 1, 0]),
    ("vector database benchmarks", [0, 0, 0, 1, 0, 0, 0]),
    ("uhh basically what is faiss vs scann i think faster", [0, 1, 0, 0, 0, 1, 1]),
    ("so like how do i tune bm25 params kinda confused", [0, 1, 0, 0, 0, 0, 1]),
]
```

(The `train()` assertion `len(y) == len(ROUTER_LABELS)` now enforces 7 automatically.)

- [ ] **Step 6: Verify trainer self-check passes (no model artifact needed for CI)**

Run: `python -c "from src.training.train_query_router import SEED_DATA; from src.internal.retrieval.query_router import ROUTER_LABELS; assert all(len(y)==len(ROUTER_LABELS) for _,y in SEED_DATA); print('seed ok', len(ROUTER_LABELS))"`
Expected: `seed ok 7`

- [ ] **Step 7: Lint + commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/retrieval/query_router.py src/training/train_query_router.py \
        tests/unit/retrieval/test_query_router.py
git commit -m "feat(router): add rewrite label, skip expensive legs on easy queries, retrain seed"
```

---

### Task 5: Acceptance — benchmark grid, eval gate, full suite, PR

Proves all three gates and ships.

**Files:**
- Run-only + a short results note appended to this plan.

- [ ] **Step 1: Full test suite green**

Run: `pytest`
Expected: PASS, count ≥ prior 2011, zero failures.

- [ ] **Step 2: Benchmark grid over candidate configs**

Run the existing benchmark harness (via its test driver or a one-off script that calls `run_query_transform_benchmark`) over configs that toggle `rewrite`, `multi_query`, `decompose`, `hyde`. Capture the sorted recall@10 table.

Run: `pytest tests/unit/retrieval/test_query_transform_benchmark.py -v`
Expected: PASS; record the winning config signature.

- [ ] **Step 3: Eval gate vs baseline**

Start a retrieval server (`python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl`) and run:

Run: `python -m src.internal.retrieval.eval_runner --dataset data/eval/qa_pairs.jsonl --top_k 10 --retrieval_url http://localhost:8001/retrieve`
Expected: recall@10 and nDCG@10 ≥ values in `data/eval/baseline_metrics.json`, with a measurable gain on ≥1 metric; p99 transform latency does not trip `qt_slo_exceeded`.

- [ ] **Step 4: Append a results note to this plan**

Add a short `## Results (M9)` section: baseline vs tuned recall@10/nDCG@10/MRR, the winning benchmark config, and the p99 latency. Commit.

```bash
git add docs/superpowers/plans/2026-06-23-query-transformations-optimization-tuning.md
git commit -m "docs(plan): record M9 eval + benchmark results"
```

- [ ] **Step 5: Push and open PR**

```bash
git push -u origin feat/query-transform-optimization-tuning
gh pr create --title "Query Transformations M9: query-rewrite leg, prompt + fusion tuning, smarter routing" \
  --body "$(cat <<'EOF'
## Summary
M9 tuning of the existing query-transform stack (no new modules):
- New canonical query-rewrite leg (`QT_REWRITE`) threaded through config/enhancer/pipeline/bundle/router.
- Tightened decompose / step-back / multi-query prompts for recall.
- Graded adaptive MMR λ (fixes ≤3-token docstring/code mismatch).
- Router now skips expensive legs on easy queries and routes `rewrite` for noisy/long ones; seed widened to 7 labels.

## Acceptance
- eval_runner recall@10/nDCG@10 ≥ baseline (see plan Results note); p99 SLO held.
- query_transform_benchmark grid justifies tuned defaults.
- Full pytest green, no regressions.

Spec: docs/superpowers/specs/2026-06-23-query-transformations-optimization-tuning-design.md
Plan: docs/superpowers/plans/2026-06-23-query-transformations-optimization-tuning.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** quality (Tasks 1–3), latency/cost (Task 4 routing = skip legs), smarter routing (Task 4), gap-fill in place (Task 1 rewrite leg). Acceptance gates all in Task 5. ✓
- **No new files:** every Modify path pre-exists. ✓
- **Type consistency:** `rewrite` is `bool` in `QueryTransformConfig`, `str | None` in `TransformedQueryBundle`, `str | None` return from `QueryEnhancer.rewrite`, and label `"rewrite"` (index 6) in `ROUTER_LABELS` — consistent across Tasks 1 and 4. ✓
- **Tradeoff noted:** Latency/cost is pursued via routing rather than speculative async/cache default changes, keeping edits evidence-driven; revisit async timeout/cache TTL only if Task 5's p99 shows a problem.
