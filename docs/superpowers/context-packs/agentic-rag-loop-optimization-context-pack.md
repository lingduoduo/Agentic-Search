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

### 6. Testing strategy

All 10 existing tests must stay green (they encode the LLM-response call order — the
changes add no extra `complete()` calls, so order is preserved). Add focused tests:

1. **`merged` init** — `AgenticRAGConfig(max_rounds=0)` → `run` returns an
   `AgenticRAGResult` with empty context, no crash.
2. **Query normalization** — initial queries differing only by case/whitespace
   (`"GPT-4 cost"` vs `"gpt-4 cost "`) trigger retrieval once.
3. **Doc dedup by content** — two retrieved docs with no URL and identical content but
   different ids collapse to one; distinct content stays separate.
4. **Follow-up cap** — gap analysis returning 8 queries retrieves ≤5 in the next round.
5. **Sufficiency timeout** — a `complete` that sleeps past `sufficiency_timeout_s`
   returns `True` (fail-open) and the loop proceeds to synthesis without hanging (use a
   small `sufficiency_timeout_s` like `0.05`).

## Implementation Plan Context

### Global Constraints

- Keep happy-path output identical; all 10 existing tests in `tests/unit/test_agentic_rag.py` must stay green.
- Retrieval always receives the **original** query string; normalization governs only the dedup set.
- Sufficiency check stays **fail-open**: on timeout/error it returns `True` (stop looping). Do NOT flip to fail-closed.
- No empty-evidence canned fallback; the loop still calls `generate_answer` with zero docs.
- Timeout applies only to the sufficiency check, not gap analysis.
- `max_followups_per_round` default = `5`; `sufficiency_timeout_s` default = `5.0`.
- Feature branch only (never commit to `main`). Run `ruff check . --fix && ruff format .` and full `pytest` before done.

---

### Task 1: Crash guard — initialize `merged` before the loop

**Files:**
- Modify: `src/agents/agentic_rag.py` (inside `run`, just before `for round_idx in range(...)`)
- Test: `tests/unit/test_agentic_rag.py`

**Interfaces:**
- Consumes: existing `AgenticRAGLoop(config, llm)`, `run(question)`.
- Produces: `run` never raises `UnboundLocalError` for `merged` when `max_rounds == 0` or no novel queries exist.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_run_with_zero_max_rounds_returns_empty_context():
    llm = _llm_responses("sub", "hyde", "broader", "answer")
    config = AgenticRAGConfig(max_rounds=0, topk=5)
    with patch(
        "src.agents.agentic_rag.retrieve_context",
        AsyncMock(return_value=_make_bundle(["d1"])),
    ):
        loop = AgenticRAGLoop(config, llm=llm)
        result = await loop.run("q?")
    assert isinstance(result, AgenticRAGResult)
    assert result.rounds_used == 0
    assert result.context.documents == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_agentic_rag.py::test_run_with_zero_max_rounds_returns_empty_context -v`
Expected: FAIL with `UnboundLocalError: local variable 'merged' referenced before assignment`

- [ ] **Step 3: Write minimal implementation**

In `run`, immediately before `for round_idx in range(self.config.max_rounds):`, add:

```python
        merged = SearchContextBundle(query=question, documents=[])

        for round_idx in range(self.config.max_rounds):
```

- [ ] **Step 4: Run test to verify it passes**

_[Section compacted.]_

### Task 2: Normalized, intra-batch query deduplication

**Files:**
- Modify: `src/agents/agentic_rag.py` (add helpers `_norm_query`, `_dedupe_novel`; rewrite the `seen_queries` bookkeeping in `run`)
- Test: `tests/unit/test_agentic_rag.py`

**Interfaces:**
- Produces:
  - `_norm_query(q: str) -> str` — lowercase + whitespace-collapsed form.
  - `_dedupe_novel(queries: list[str], seen: set[str]) -> list[str]` — returns items whose normalized form is not yet in `seen`, recording each into `seen` as it goes (dedupes within the batch and across rounds). Returned strings are the **original** (un-normalized) queries.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_case_and_whitespace_variants_retrieve_once():
    bundle = _make_bundle(["d1"])
    # decompose/hyde/step_back produce case/whitespace variants of one query
    llm = _llm_responses(
        "GPT-4 cost",       # decompose
        "gpt-4   cost ",    # hyde (normalizes to same as decompose)
        " GPT-4 Cost",      # step_back (same normalized form)
        "yes",              # sufficiency
        "answer",           # generate_answer
    )
    config = AgenticRAGConfig(max_rounds=3, topk=5)
    calls: list[str] = []

    async def _track(query, **kwargs):
        calls.append(query)
        return bundle

    with patch("src.agents.agentic_rag.retrieve_context", side_effect=_track):
        loop = AgenticRAGLoop(config, llm=llm)
        await loop.run("gpt-4 cost?")

    # all three enhanced queries normalize identically → retrieved once
    assert len(calls) == 1
```

- [ ] **Step 2: Run test to verify it fails**

_[Section compacted.]_

### Task 3: Robust document dedup key

**Files:**
- Modify: `src/agents/agentic_rag.py` (add `import hashlib`; add `_doc_key`; use it in the accumulate loop)
- Test: `tests/unit/test_agentic_rag.py`

**Interfaces:**
- Produces: `_doc_key(doc: ContextDocument) -> str` — normalized URL (`strip().lower()`) when a URL is present, else the SHA-256 hex digest of the full content.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_url_less_docs_dedup_by_full_content():
    # two docs, no URL, identical content but different ids → one doc;
    # a third with different content → kept.
    dup_a = ContextDocument(id="a", title="A", content="X" * 200, score=0.9)
    dup_b = ContextDocument(id="b", title="B", content="X" * 200, score=0.8)
    other = ContextDocument(id="c", title="C", content="Y" * 200, score=0.7)
    bundle = SearchContextBundle(query="q", documents=[dup_a, dup_b, other])
    config = AgenticRAGConfig(max_rounds=1, topk=5)
    with patch(
        "src.agents.agentic_rag.retrieve_context", AsyncMock(return_value=bundle)
    ):
        loop = AgenticRAGLoop(config, llm=None)
        result = await loop.run("q?")
    # dup_a/dup_b collapse; other stays → 2 unique docs
    assert len(result.context.documents) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_agentic_rag.py::test_url_less_docs_dedup_by_full_content -v`
Expected: PASS-BY-ACCIDENT risk — with the current `content[:120]` key the two `"X"*200` docs already share a prefix, so this specific case passes today. Adjust the test to prove the collision the hash fixes:

_[Section compacted.]_

### Task 4: Cap follow-up queries per round

**Files:**
- Modify: `src/agents/agentic_rag.py` (`AgenticRAGConfig` field + truncate `novel_follow_ups`)
- Test: `tests/unit/test_agentic_rag.py`

**Interfaces:**
- Consumes: `_dedupe_novel` from Task 2.
- Produces: `AgenticRAGConfig.max_followups_per_round: int = 5`; at most that many queries are retrieved in the round following a gap analysis.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_follow_ups_capped_per_round():
    bundle = _make_bundle(["d1"])
    eight = "\n".join(f"followup query {i}" for i in range(8))
    llm = _llm_responses(
        "sub", "hyde", "broader",              # enhance
        "no",                                  # sufficiency round 1
        f"GAPS:\ng\nQUERIES:\n{eight}",        # 8 follow-ups
        "yes",                                 # sufficiency round 2
        "answer",                              # generate_answer
    )
    config = AgenticRAGConfig(max_rounds=3, topk=5, max_followups_per_round=5)
    calls: list[str] = []

    async def _track(query, **kwargs):
        calls.append(query)
        return bundle

    with patch("src.agents.agentic_rag.retrieve_context", side_effect=_track):
        loop = AgenticRAGLoop(config, llm=llm)
        await loop.run("q?")

    followup_calls = [c for c in calls if c.startswith("followup query")]
    assert len(followup_calls) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_agentic_rag.py::test_follow_ups_capped_per_round -v`

_[Section compacted.]_

### Task 5: Single-pass `_clean_line` in `_parse_gap_queries`

**Files:**
- Modify: `src/agents/agentic_rag.py` (`_parse_gap_queries` return)
- Test: covered by existing `test_follow_up_queries_do_not_duplicate_seen_queries` + full suite (pure refactor, no behavior change)

**Interfaces:**
- Produces: `_parse_gap_queries` output unchanged; each line cleaned once instead of twice.

- [ ] **Step 1: Write minimal implementation** (refactor — no new test; behavior identical)

Replace:

```python
    return [
        _clean_line(line) for line in queries_section.splitlines() if _clean_line(line)
    ]
```

with:

```python
    queries: list[str] = []
    for line in queries_section.splitlines():
        cleaned = _clean_line(line)
        if cleaned:
            queries.append(cleaned)
    return queries
```

- [ ] **Step 2: Run tests to verify no regression**

Run: `pytest tests/unit/test_agentic_rag.py -v`
Expected: PASS (all, including gap-analysis tests)

- [ ] **Step 3: Commit**

```bash
git add src/agents/agentic_rag.py
git commit -m "refactor(agentic-rag): clean each gap-query line once"
```

---

### Task 6: Bounded timeout on the sufficiency check

**Files:**
- Modify: `src/agents/agentic_rag.py` (`import asyncio`; `AgenticRAGConfig` field; `_is_sufficient` → async with `wait_for`/`to_thread`; `await` at call site)
- Test: `tests/unit/test_agentic_rag.py`

**Interfaces:**
- Produces:
  - `AgenticRAGConfig.sufficiency_timeout_s: float = 5.0`.
  - `_is_sufficient(self, question, context) -> bool` is now a coroutine; awaited in `run`.
  - On `asyncio.TimeoutError` or any exception → logs a warning and returns `True` (fail-open, unchanged).

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_sufficiency_check_times_out_fail_open():
    import time as _time

    def _slow_complete(messages, **kwargs):
        _time.sleep(1.0)  # exceeds the tiny timeout
        return "no"

    llm = MagicMock()
    llm.complete.side_effect = _slow_complete
    config = AgenticRAGConfig(sufficiency_timeout_s=0.05)
    loop = AgenticRAGLoop(config, llm=llm)
    bundle = _make_bundle(["d1"])

    result = await loop._is_sufficient("q?", bundle)
    assert result is True  # fail-open on timeout, and returns promptly
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_agentic_rag.py::test_sufficiency_check_times_out_fail_open -v`
Expected: FAIL — `_is_sufficient` is currently sync (`assert <coroutine> ... ` / `TypeError`) and has no timeout, so awaiting it or the call fails.

- [ ] **Step 3: Write minimal implementation**

Add `import asyncio` to the stdlib import block. Add the config field:

```python
@dataclass(frozen=True)
class AgenticRAGConfig:
    max_rounds: int = 3

_[Section compacted.]_

### Task 7: Lint, full-suite gate, and PR prep

**Files:** none (verification + docs already committed)

- [ ] **Step 1: Lint and format**

Run: `ruff check . --fix && ruff format .`
Expected: clean (commit any formatting deltas with `style(agentic-rag): ruff format`).

- [ ] **Step 2: Full unit + regression gate**

Run: `pytest`
Expected: PASS (all suites green; no regressions elsewhere).

- [ ] **Step 3: Push branch and open PR**

Ensure the spec (docs/superpowers/specs/2026-07-01-agentic-rag-loop-optimization-design.md) and this plan are committed on the branch, then push and open a PR with a unique, specific title.

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
