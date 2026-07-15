# Agent Framework Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply one targeted optimization to each of the five search-loop components (Planner, Search Tool, Reranker Tool, Evidence Judge, Answer Generator) plus one zero-default GRPO reward term and one opt-in loop metric — improving answer quality, cost/latency, GRPO reward shaping, and robustness without a refactor.

**Architecture:** The five components in `src/agents/components/` are the standalone, independently-tested component API (production `SearchAgentLoop` consumes only `EvidenceJudge.score_round`). Each task edits one component plus its tests in `tests/unit/test_components.py`. Two further tasks add the `early_stop_bonus` reward term (`src/training/reward.py`) and an opt-in `evidence_plateau_min_gain` detection metric (`src/agents/search.py`). All changes are backward-compatible: new args are optional, the new reward weight defaults to 0, and the new loop metric is off unless explicitly configured.

**Tech Stack:** Python 3.11+, dataclasses + async (existing style), `pytest`, `ruff`. No new dependencies.

## Global Constraints

- **Never commit to `main`** — work on branch `feat/agent-framework-optimization` (already created); open a PR at the end.
- **New component args are optional** — existing call sites and tests must pass unchanged.
- **New reward weight defaults to `0.0`** — `sparse_final_only`, `second_pass`, `third_pass_with_format`, and `retriever_aware` totals must be byte-identical when the weight is 0.
- **New loop metric is opt-in** — `evidence_plateau_min_gain` defaults to `None`; with `None` the loop is byte-identical and `early_stops` stays `0.0`.
- **Degrade-don't-crash** on backend failure (log a warning, fall back) — matches the existing pattern.
- **Run `pytest` + `ruff check . --fix && ruff format .` before every commit.**
- **`SearchResult`** fields used in tests: `SearchResult(contents: str, score: float, title: str | None)` (from `src/context/search.py`).
- Commit message trailer on every commit: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

### Task 1: Planner — duplicate-query guard + bounded fallback

**Files:**
- Modify: `src/agents/components/planner.py`
- Test: `tests/unit/test_components.py`

**Interfaces:**
- Consumes: `Retriever` enum (`src/agents/state.py`).
- Produces: `SearchAction(query: str, retriever: Retriever, is_duplicate: bool = False)`; `Planner.decide(self, text: str, previous_queries: Sequence[str] = ()) -> PlannerDecision`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_components.py` (Planner section):

```python
def test_planner_flags_duplicate_query() -> None:
    from src.agents.components.planner import Planner, SearchAction

    decision = Planner().decide(
        "<search>what is faiss</search>", previous_queries=["what is faiss"]
    )

    assert isinstance(decision, SearchAction)
    assert decision.query == "what is faiss"
    assert decision.is_duplicate is True


def test_planner_new_query_not_flagged_duplicate() -> None:
    from src.agents.components.planner import Planner, SearchAction

    decision = Planner().decide(
        "<search>brand new</search>", previous_queries=["what is faiss"]
    )

    assert isinstance(decision, SearchAction)
    assert decision.is_duplicate is False


def test_planner_duplicate_match_ignores_whitespace_and_case() -> None:
    from src.agents.components.planner import Planner

    decision = Planner().decide(
        "<search>  What  Is   FAISS </search>", previous_queries=["what is faiss"]
    )

    assert decision.is_duplicate is True


def test_planner_fallback_query_is_bounded() -> None:
    from src.agents.components.planner import Planner, SearchAction

    raw = "first line of reasoning\n" + ("x" * 1000)
    decision = Planner().decide(raw)

    assert isinstance(decision, SearchAction)
    assert decision.query == "first line of reasoning"
    assert len(decision.query) <= 256
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_components.py -k "planner_flags_duplicate or planner_new_query_not or planner_duplicate_match or planner_fallback_query" -v`
Expected: FAIL (`decide()` got an unexpected keyword argument `previous_queries`, or `is_duplicate` missing).

- [ ] **Step 3: Implement the change**

In `src/agents/components/planner.py`:

Add the import at the top (after `import re`):

```python
from collections.abc import Sequence
```

Add the field to `SearchAction`:

```python
@dataclass(frozen=True)
class SearchAction:
    query: str
    retriever: Retriever = Retriever.VECTOR_DB
    is_duplicate: bool = False
```

Add a module-level helper and a fallback bound constant (after `_RETRIEVER_BY_NAME`):

```python
_FALLBACK_QUERY_MAX_CHARS = 256


def _normalize_query(query: str) -> str:
    """Whitespace- and case-insensitive key for duplicate detection."""
    return " ".join(query.split()).casefold()
```

Replace `Planner.decide` with:

```python
    def decide(
        self, text: str, previous_queries: Sequence[str] = ()
    ) -> PlannerDecision:
        seen = {_normalize_query(q) for q in previous_queries}
        search = _SEARCH_RE.search(text)
        if search:
            retriever = _RETRIEVER_BY_NAME.get(
                (search.group("retriever") or "").lower(), Retriever.VECTOR_DB
            )
            query = search.group("query").strip()
            return SearchAction(
                query=query,
                retriever=retriever,
                is_duplicate=_normalize_query(query) in seen,
            )

        if _RERANK_RE.search(text):
            return RerankAction()

        answer = _ANSWER_RE.search(text)
        if answer:
            return AnswerAction(text=answer.group("text").strip())

        # Safe default: a *bounded* best-effort vector-DB search on the first
        # non-empty line, so a long reasoning trace is never dumped at the retriever.
        fallback = next(
            (line.strip() for line in text.splitlines() if line.strip()), ""
        )[:_FALLBACK_QUERY_MAX_CHARS]
        return SearchAction(
            query=fallback,
            retriever=Retriever.VECTOR_DB,
            is_duplicate=_normalize_query(fallback) in seen,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_components.py -k planner -v`
Expected: PASS (new + all existing planner tests).

- [ ] **Step 5: Lint and commit**

```bash
ruff check src/agents/components/planner.py tests/unit/test_components.py --fix && ruff format src/agents/components/planner.py tests/unit/test_components.py
git add src/agents/components/planner.py tests/unit/test_components.py
git commit -m "feat: planner duplicate-query guard and bounded fallback

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Search Tool — per-instance result cache + web-exception degradation

**Files:**
- Modify: `src/agents/components/search_tool.py`
- Test: `tests/unit/test_components.py`

**Interfaces:**
- Consumes: `Retriever`, `SearchAgentState` (`src/agents/state.py`), `SearchResult`.
- Produces: unchanged `SearchTool(vector_db_fn, web_fn=None)` and `async run(state, query, retriever=Retriever.VECTOR_DB) -> list[SearchResult]`; internally caches by `(retriever, normalized_query)` and degrades web→vdb on exception.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_components.py` (SearchTool section):

```python
async def test_search_tool_caches_repeated_query() -> None:
    from src.agents.components.search_tool import SearchTool

    calls = {"n": 0}

    async def vdb(query: str) -> list[SearchResult]:
        calls["n"] += 1
        return [_result(title="vdb")]

    tool = SearchTool(vdb)
    state = SearchAgentState(question="q")
    await tool.run(state, "faiss index")
    await tool.run(state, "  FAISS   index ")  # same query, different spacing/case

    assert calls["n"] == 1  # second call served from cache, no extra backend hit


async def test_search_tool_caches_per_backend() -> None:
    from src.agents.components.search_tool import SearchTool

    vdb_calls = {"n": 0}
    web_calls = {"n": 0}

    async def vdb(query: str) -> list[SearchResult]:
        vdb_calls["n"] += 1
        return [_result(title="vdb")]

    async def web(query: str) -> list[SearchResult]:
        web_calls["n"] += 1
        return [_result(title="web")]

    tool = SearchTool(vdb, web_fn=web)
    state = SearchAgentState(question="q")
    await tool.run(state, "q", retriever=Retriever.VECTOR_DB)
    await tool.run(state, "q", retriever=Retriever.WEB)  # same text, other backend

    assert vdb_calls["n"] == 1
    assert web_calls["n"] == 1  # different backend → not a cache hit


async def test_search_tool_degrades_to_vdb_when_web_raises() -> None:
    from src.agents.components.search_tool import SearchTool

    async def vdb(query: str) -> list[SearchResult]:
        return [_result(title="vdb")]

    async def web(query: str) -> list[SearchResult]:
        raise RuntimeError("web backend exploded")

    state = SearchAgentState(question="q")
    docs = await SearchTool(vdb, web_fn=web).run(state, "q", retriever=Retriever.WEB)

    assert [d.title for d in docs] == ["vdb"]
    assert state.search_rounds == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_components.py -k "search_tool_caches or search_tool_degrades_to_vdb_when_web_raises" -v`
Expected: FAIL (`calls["n"] == 2` for the cache test; `RuntimeError` propagates for the raise test).

- [ ] **Step 3: Implement the change**

Replace the body of `src/agents/components/search_tool.py` (keep the module docstring and imports; add `Retriever` is already imported) with this class:

```python
class SearchTool:
    """Run one retriever call and fold the results into the agent state.

    Caches results per ``(backend, normalized query)`` for the lifetime of the
    instance (one question), so a repeated query never re-hits the backend. A
    web call that *raises* degrades to the vector-DB backend, matching the
    "web unconfigured" degradation already in place.
    """

    def __init__(
        self, vector_db_fn: RetrieveFn, web_fn: RetrieveFn | None = None
    ) -> None:
        self._vector_db_fn = vector_db_fn
        self._web_fn = web_fn
        self._cache: dict[tuple[Retriever, str], list[SearchResult]] = {}

    async def run(
        self,
        state: SearchAgentState,
        query: str,
        retriever: Retriever = Retriever.VECTOR_DB,
    ) -> list[SearchResult]:
        key = (retriever, " ".join(query.split()).casefold())
        if key in self._cache:
            docs = self._cache[key]
        else:
            docs = await self._retrieve(retriever, query)
            self._cache[key] = docs
        state.record_search(query, docs)
        return docs

    async def _retrieve(
        self, retriever: Retriever, query: str
    ) -> list[SearchResult]:
        if retriever is Retriever.WEB and self._web_fn is not None:
            try:
                return await self._web_fn(query)
            except Exception:  # noqa: BLE001 - degrade-don't-crash on web failure
                logger.warning(
                    "Web retriever raised; degrading to vector-DB backend.",
                    exc_info=True,
                )
                return await self._vector_db_fn(query)
        if retriever is Retriever.WEB:
            logger.warning(
                "Web retriever requested but not configured; "
                "degrading to vector-DB backend."
            )
        return await self._vector_db_fn(query)
```

(Delete the old `_select` method — it is replaced by `_retrieve`. `RetrieveFn` and `logger` are already defined at module top.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_components.py -k search_tool -v`
Expected: PASS (new + all existing SearchTool tests, including the existing unconfigured-degradation test).

- [ ] **Step 5: Lint and commit**

```bash
ruff check src/agents/components/search_tool.py tests/unit/test_components.py --fix && ruff format src/agents/components/search_tool.py tests/unit/test_components.py
git add src/agents/components/search_tool.py tests/unit/test_components.py
git commit -m "feat: search tool result cache and web-exception degradation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Reranker Tool — bounded candidate window + ≤1-doc skip

**Files:**
- Modify: `src/agents/components/reranker_tool.py`
- Test: `tests/unit/test_components.py`

**Interfaces:**
- Consumes: `SearchAgentState`, `SearchResult`, `RerankFn = Callable[[str, list[SearchResult]], list[SearchResult]]`.
- Produces: `RerankerTool(rerank_fn, max_candidates: int | None = None)`; unchanged `run(state, query=None) -> list[SearchResult]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_components.py` (RerankerTool section):

```python
def test_reranker_tool_skips_when_single_doc() -> None:
    from src.agents.components.reranker_tool import RerankerTool

    called = {"n": 0}

    def fake_rerank(query: str, docs: list[SearchResult]) -> list[SearchResult]:
        called["n"] += 1
        return docs

    state = SearchAgentState(question="q")
    state.record_search("q", [_result(title="only")])

    reordered = RerankerTool(fake_rerank).run(state)

    assert called["n"] == 0  # nothing to reorder with one doc
    assert [d.title for d in reordered] == ["only"]


def test_reranker_tool_limits_to_max_candidates() -> None:
    from src.agents.components.reranker_tool import RerankerTool

    seen_lengths: list[int] = []

    def fake_rerank(query: str, docs: list[SearchResult]) -> list[SearchResult]:
        seen_lengths.append(len(docs))
        return list(reversed(docs))

    state = SearchAgentState(question="q")
    state.record_search(
        "q",
        [_result(title="a"), _result(title="b"), _result(title="c"), _result(title="d")],
    )

    reordered = RerankerTool(fake_rerank, max_candidates=2).run(state)

    assert seen_lengths == [2]  # only the top-2 were handed to the reranker
    # top-2 [a, b] reversed -> [b, a], tail [c, d] preserved
    assert [d.title for d in reordered] == ["b", "a", "c", "d"]
    assert [d.title for d in state.retrieved_docs] == ["b", "a", "c", "d"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_components.py -k "reranker_tool_skips_when_single or reranker_tool_limits" -v`
Expected: FAIL (`max_candidates` is not a valid argument; single-doc path still calls `fake_rerank`).

- [ ] **Step 3: Implement the change**

Replace the class in `src/agents/components/reranker_tool.py` with:

```python
class RerankerTool:
    """Re-order ``state.retrieved_docs`` in place using the rerank function.

    ``max_candidates`` bounds cost: only the top-N docs (by current order) are
    scored by the cross-encoder; the rest keep their position. ``None`` (default)
    reranks the whole set. A set of <= 1 doc is a guaranteed no-op, so the
    reranker is not called at all.
    """

    def __init__(self, rerank_fn: RerankFn, max_candidates: int | None = None) -> None:
        self._rerank_fn = rerank_fn
        self._max_candidates = max_candidates

    def run(
        self, state: SearchAgentState, query: str | None = None
    ) -> list[SearchResult]:
        docs = list(state.retrieved_docs)
        if len(docs) <= 1:
            return docs
        if self._max_candidates is None or self._max_candidates >= len(docs):
            reordered = self._rerank_fn(query or state.question, docs)
        else:
            head = docs[: self._max_candidates]
            tail = docs[self._max_candidates :]
            reordered = self._rerank_fn(query or state.question, head) + tail
        state.record_rerank(reordered)
        return reordered
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_components.py -k reranker -v`
Expected: PASS (new + existing reranker tests; the existing empty-docs and full-reorder tests still pass).

- [ ] **Step 5: Lint and commit**

```bash
ruff check src/agents/components/reranker_tool.py tests/unit/test_components.py --fix && ruff format src/agents/components/reranker_tool.py tests/unit/test_components.py
git add src/agents/components/reranker_tool.py tests/unit/test_components.py
git commit -m "feat: reranker bounded candidate window and single-doc skip

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Evidence Judge — marginal-gain + should_stop plateau helpers

**Files:**
- Modify: `src/agents/components/evidence_judge.py`
- Test: `tests/unit/test_components.py`

**Interfaces:**
- Produces (both static, pure): `EvidenceJudge.marginal_gain(prev: float, curr: float) -> float`; `EvidenceJudge.should_stop(prev: float, curr: float, min_gain: float) -> bool`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_components.py` (EvidenceJudge section):

```python
def test_evidence_judge_marginal_gain_is_delta() -> None:
    from src.agents.components.evidence_judge import EvidenceJudge

    assert EvidenceJudge.marginal_gain(0.4, 0.7) == 0.3
    assert EvidenceJudge.marginal_gain(0.7, 0.7) == 0.0


def test_evidence_judge_should_stop_on_plateau() -> None:
    from src.agents.components.evidence_judge import EvidenceJudge

    # gain 0.01 < min_gain 0.05 -> plateau -> stop
    assert EvidenceJudge.should_stop(0.70, 0.71, min_gain=0.05) is True


def test_evidence_judge_should_not_stop_on_real_gain() -> None:
    from src.agents.components.evidence_judge import EvidenceJudge

    # gain 0.20 >= min_gain 0.05 -> keep searching
    assert EvidenceJudge.should_stop(0.50, 0.70, min_gain=0.05) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_components.py -k "marginal_gain or should_stop or should_not_stop" -v`
Expected: FAIL (`AttributeError: type object 'EvidenceJudge' has no attribute 'marginal_gain'`).

- [ ] **Step 3: Implement the change**

In `src/agents/components/evidence_judge.py`, add two static methods to the `EvidenceJudge` class (place them next to `score_round`):

```python
    @staticmethod
    def marginal_gain(prev: float, curr: float) -> float:
        """Improvement in evidence_score from one round to the next."""
        return curr - prev

    @staticmethod
    def should_stop(prev: float, curr: float, min_gain: float) -> bool:
        """True when the latest round's gain falls below ``min_gain`` (plateau).

        A plateau means another search round is unlikely to help, so the policy
        (or an opt-in loop) can choose to stop searching. This never forces an
        answer; it only signals that more searching has diminishing returns.
        """
        return EvidenceJudge.marginal_gain(prev, curr) < min_gain
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_components.py -k "marginal_gain or should_stop or should_not_stop" -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
ruff check src/agents/components/evidence_judge.py tests/unit/test_components.py --fix && ruff format src/agents/components/evidence_judge.py tests/unit/test_components.py
git add src/agents/components/evidence_judge.py tests/unit/test_components.py
git commit -m "feat: evidence judge marginal-gain and plateau should_stop helpers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Answer Generator — appearance-ordered, de-duplicated citations

**Files:**
- Modify: `src/agents/components/answer_generator.py`
- Test: `tests/unit/test_components.py`

**Interfaces:**
- Consumes: `AgentContext` (`src/context/search.py`), `Citation`, `SearchAgentState`.
- Produces: unchanged `AnswerGenerator.generate(answer_text, ctx) -> AnswerResult`; citations now ordered by first appearance in `answer_text` and de-duplicated by doc contents.

- [ ] **Step 1: Write the failing tests**

Add these helpers + tests to `tests/unit/test_components.py` (AnswerGenerator section). The two-round context lets us reference markers out of retrieval order:

```python
def _ctx_two_rounds_dup() -> AgentContext:
    ctx = AgentContext()
    ctx.add_round(["q1"], [[_result(text="alpha body", title="A")]])
    # Second round re-retrieves the same "alpha body" plus a new doc.
    ctx.add_round(
        ["q2"],
        [[_result(text="alpha body", title="A"), _result(text="gamma body", title="G")]],
    )
    return ctx


def test_answer_generator_orders_citations_by_appearance() -> None:
    from src.agents.components.answer_generator import AnswerGenerator

    ctx = _ctx_two_rounds_dup()
    # Reference the round-2 doc first, then the round-1 doc.
    answer = "See gamma [R2Q1D2] and also alpha [R1Q1D1]."
    result = AnswerGenerator().generate(answer, ctx)

    assert [c.doc_id for c in result.citations] == ["R2Q1D2", "R1Q1D1"]


def test_answer_generator_collapses_duplicate_doc_contents() -> None:
    from src.agents.components.answer_generator import AnswerGenerator

    ctx = _ctx_two_rounds_dup()
    # Both R1Q1D1 and R2Q1D1 point at identical "alpha body".
    answer = "Alpha first [R1Q1D1], alpha again [R2Q1D1]."
    result = AnswerGenerator().generate(answer, ctx)

    # Same contents -> a single citation, keyed by the first-cited marker.
    assert [c.doc_id for c in result.citations] == ["R1Q1D1"]
    assert [c.text for c in result.citations] == ["alpha body"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_components.py -k "orders_citations_by_appearance or collapses_duplicate" -v`
Expected: FAIL (citations come back in retrieval order / with duplicate contents).

- [ ] **Step 3: Implement the change**

Replace `AnswerGenerator.generate` in `src/agents/components/answer_generator.py` with:

```python
    def generate(self, answer_text: str, ctx: AgentContext) -> AnswerResult:
        valid_keys = ctx.cited_result_ids(answer_text)
        # Collect every valid (key, contents) the answer references.
        candidates: list[tuple[str, str]] = []
        for round_idx, round_ctxs in enumerate(ctx.rounds, 1):
            for q_idx, search_ctx in enumerate(round_ctxs, 1):
                for d_idx, result in enumerate(search_ctx.results, 1):
                    key = f"R{round_idx}Q{q_idx}D{d_idx}"
                    if key in valid_keys:
                        candidates.append((key, result.contents))

        # Order by first appearance of the marker in the answer text, then drop
        # later citations whose contents already appeared (collapse duplicates).
        candidates.sort(key=lambda kc: answer_text.find(f"[{kc[0]}]"))
        seen_contents: set[str] = set()
        citations: list[Citation] = []
        for key, contents in candidates:
            if contents in seen_contents:
                continue
            seen_contents.add(contents)
            citations.append(
                Citation(doc_id=key, marker=f"[{key}]", text=contents)
            )
        return AnswerResult(answer=answer_text, citations=citations)
```

(`update_state` is unchanged — it calls `generate` and writes `result.citations`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_components.py -k answer_generator -v`
Expected: PASS (new + all existing AnswerGenerator tests — the single-round tests still produce the same single citation).

- [ ] **Step 5: Lint and commit**

```bash
ruff check src/agents/components/answer_generator.py tests/unit/test_components.py --fix && ruff format src/agents/components/answer_generator.py tests/unit/test_components.py
git add src/agents/components/answer_generator.py tests/unit/test_components.py
git commit -m "feat: answer generator appearance-ordered de-duplicated citations

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Reward — `early_stop_bonus` term (zero-default)

**Files:**
- Modify: `src/training/reward.py`
- Test: `tests/unit/test_reward.py`

**Interfaces:**
- Consumes: `output.metrics["early_stops"]` (a float, default 0 when absent).
- Produces: `SearchRewardConfig.early_stop_bonus: float = 0.0`; reward breakdown gains an `"early_stop_bonus"` component; `retriever_aware(..., early_stop_bonus=0.05)` surfaces it.

- [ ] **Step 1: Write the failing tests**

First, read an existing test in `tests/unit/test_reward.py` to match the construction pattern for `AgentLoopOutput` / `compute`. Then add:

```python
def test_early_stop_bonus_zero_by_default_is_inert() -> None:
    from src.training.reward import SearchRewardConfig

    cfg = SearchRewardConfig.second_pass()
    assert cfg.early_stop_bonus == 0.0


def test_retriever_aware_surfaces_early_stop_bonus() -> None:
    from src.training.reward import SearchRewardConfig

    cfg = SearchRewardConfig.retriever_aware(early_stop_bonus=0.05)
    assert cfg.early_stop_bonus == 0.05
```

Add a breakdown test mirroring the existing `_breakdown`/`compute` style already used in the file (use the same `AgentLoopOutput` factory the neighboring tests use). The assertion that matters:

```python
def test_early_stop_bonus_rewards_early_stops_metric() -> None:
    from src.training.reward import SearchRewardConfig, SearchRewardFunction

    cfg = SearchRewardConfig.retriever_aware(early_stop_bonus=0.1)
    fn = SearchRewardFunction(cfg)
    # Build a metrics dict with two plateau rounds flagged.
    components = fn._reward_components_from_parts(  # match the helper used elsewhere
        correctness=0.0,
        output=_make_output(metrics={"early_stops": 2.0}),
    )
    assert components["early_stop_bonus"] == 0.2
```

> NOTE TO IMPLEMENTER: the exact private helper name and `_make_output` factory differ per file — open `tests/unit/test_reward.py`, find the pattern the existing `retriever_aware` / `evidence_gain` tests use to drive a metrics dict through the reward, and copy it verbatim for this test. The assertion (`early_stop_bonus == early_stops * weight`) is the invariant.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_reward.py -k "early_stop" -v`
Expected: FAIL (`SearchRewardConfig` has no field `early_stop_bonus`).

- [ ] **Step 3: Implement the change**

In `src/training/reward.py`:

(a) Add the field after `evidence_gain_weight` (around line 252):

```python
    # Flat bonus per round flagged as an evidence plateau (early-stop signal).
    # 0.0 by default → existing presets are byte-stable.
    early_stop_bonus: float = 0.0
```

(b) Add it to `_zeroed` (in the explicit-zero block, after `evidence_gain_weight=0.0`):

```python
            early_stop_bonus=0.0,
```

(c) Add the parameter + wiring to `retriever_aware`. Change the signature to add `early_stop_bonus: float = 0.0` and the `replace(...)` call to include `early_stop_bonus=early_stop_bonus`:

```python
    @classmethod
    def retriever_aware(
        cls,
        *,
        correctness_weight: float = 1.0,
        retriever_cost_vdb: float = -0.01,
        retriever_cost_web: float = -0.05,
        rerank_cost: float = -0.02,
        evidence_gain_weight: float = 0.1,
        early_stop_bonus: float = 0.0,
    ) -> "SearchRewardConfig":
        ...
        return replace(
            cls.second_pass(correctness_weight=correctness_weight),
            retriever_cost_vdb=retriever_cost_vdb,
            retriever_cost_web=retriever_cost_web,
            rerank_cost=rerank_cost,
            evidence_gain_weight=evidence_gain_weight,
            early_stop_bonus=early_stop_bonus,
        )
```

(d) In the reward computation (the `_reward_components`-style method, after term 14 "Evidence gain" around line 688), add:

```python
        # 15. Early-stop bonus: reward rounds the loop flagged as an evidence
        # plateau (opt-in metric; 0 unless evidence_plateau_min_gain is set).
        early_stop = cfg.early_stop_bonus * metrics.get("early_stops", 0.0)
```

(e) Add `+ early_stop` to the `shaping_total` sum (after `+ evidence_gain`):

```python
            + evidence_gain
            + early_stop
```

(f) Add it to the `components` dict (after `"evidence_gain": evidence_gain,`):

```python
            "early_stop_bonus": early_stop,
```

- [ ] **Step 4: Run the tests to verify they pass + regression**

Run: `pytest tests/unit/test_reward.py -v`
Expected: PASS — new `early_stop` tests pass AND every existing preset/regression test is unchanged (the new weight is 0 everywhere except where the test sets it).

- [ ] **Step 5: Lint and commit**

```bash
ruff check src/training/reward.py tests/unit/test_reward.py --fix && ruff format src/training/reward.py tests/unit/test_reward.py
git add src/training/reward.py tests/unit/test_reward.py
git commit -m "feat: early_stop_bonus reward term (zero-default, surfaced in retriever_aware)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Loop — opt-in `evidence_plateau_min_gain` early-stop metric

**Files:**
- Modify: `src/agents/search.py` (config dataclass; `_initial_metrics`; round-scoring site near line 846)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: `EvidenceJudge.should_stop` (Task 4).
- Produces: `SearchAgentLoopConfig.evidence_plateau_min_gain: float | None = None`; `metrics["early_stops"]` initialized to `0.0` and incremented per plateau round when the config is set.

- [ ] **Step 1: Read the surrounding code**

Open `src/agents/search.py`:
- Find `SearchAgentLoopConfig` (around lines 169-229) to add the new field.
- Find `_initial_metrics` (the dict that contains `"web_searches": 0.0, ... "evidence_gain_total": 0.0` near lines 357-363) to register `"early_stops": 0.0`.
- Find the round-scoring block (lines 846-849) where `round_score`, `prev_score` are computed.

- [ ] **Step 2: Write the failing tests**

Open `tests/unit/test_agent_loop.py`, find the existing helper that builds a `SearchAgentLoop` with a fake retriever and runs it (the tests that assert on `output.metrics`). Reusing that harness, add two tests. Replace `make_loop(...)` / `run_loop(...)` below with the file's actual harness names:

```python
async def test_loop_emits_early_stops_when_plateau_configured() -> None:
    # A retriever that returns the SAME strong docs every round → evidence
    # score plateaus after round 1, so round 2's gain is below min_gain.
    loop = make_loop(  # existing harness; add the new config kwarg
        evidence_plateau_min_gain=0.05,
        # ...whatever scripted model output drives >=2 search rounds...
    )
    output = await run_loop(loop)
    assert output.metrics["early_stops"] >= 1.0


async def test_loop_early_stops_zero_by_default() -> None:
    loop = make_loop()  # no evidence_plateau_min_gain → default None
    output = await run_loop(loop)
    assert output.metrics["early_stops"] == 0.0
```

> NOTE TO IMPLEMENTER: `test_agent_loop.py` already drives multi-round searches with a scripted fake model and a fake retriever. Copy the closest existing multi-round test's setup; the only new inputs are (1) passing `evidence_plateau_min_gain=0.05` into the loop config and (2) a retriever whose repeated results cause a flat evidence score. The default-case test can reuse any existing multi-round fixture unchanged and just assert `early_stops == 0.0`.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/unit/test_agent_loop.py -k early_stop -v`
Expected: FAIL (`evidence_plateau_min_gain` is not a config field / `early_stops` KeyError).

- [ ] **Step 4: Implement the change**

(a) In `SearchAgentLoopConfig`, add the field (with the other optional tuning fields):

```python
    # When set, count rounds whose evidence_score gain falls below this value as
    # an "early_stops" metric (a plateau signal the reward can price). None
    # (default) disables it entirely — the metric stays 0 and behavior is
    # byte-identical. This is observability only; it does not terminate the loop.
    evidence_plateau_min_gain: float | None = None
```

(b) In `_initial_metrics`, add to the dict (next to `"evidence_gain_total": 0.0`):

```python
            "early_stops": 0.0,
```

(c) At the round-scoring site (currently lines 846-849), after `metrics["evidence_score_final"] = round_score`, insert:

```python
            if cfg.evidence_plateau_min_gain is not None and EvidenceJudge.should_stop(
                prev_score, round_score, cfg.evidence_plateau_min_gain
            ):
                metrics["early_stops"] += 1.0
```

(`prev_score` and `round_score` are already in scope; `EvidenceJudge` is already imported at line 30; `cfg` is the loop config in scope.)

- [ ] **Step 5: Run the tests to verify they pass + full loop regression**

Run: `pytest tests/unit/test_agent_loop.py -v`
Expected: PASS — new early-stop tests pass; every existing loop test is unchanged (default `None` path is byte-identical).

- [ ] **Step 6: Lint and commit**

```bash
ruff check src/agents/search.py tests/unit/test_agent_loop.py --fix && ruff format src/agents/search.py tests/unit/test_agent_loop.py
git add src/agents/search.py tests/unit/test_agent_loop.py
git commit -m "feat: opt-in evidence_plateau_min_gain early_stops loop metric

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Full-suite gate + PR

**Files:** none (verification + PR).

- [ ] **Step 1: Run the full unit suite**

Run: `pytest tests/unit -q`
Expected: PASS, no test-count regression vs. `main` (new tests added, none removed).

- [ ] **Step 2: Confirm the GRPO smoke test still passes**

Run: `pytest tests/unit/test_search_agent_grpo_trainer.py -v`
Expected: PASS (`test_grpo_smoke_step_with_retriever_aware_reward`).

- [ ] **Step 3: Lint + format the whole tree**

Run: `ruff check . --fix && ruff format .`
Expected: clean.

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin feat/agent-framework-optimization
gh pr create --title "Optimize agent framework components (planner, search, reranker, judge, answer) + GRPO early-stop reward" --body "$(cat <<'EOF'
## Summary
Targeted optimization of the five search-loop components plus one GRPO reward term and one opt-in loop metric. One change per component; all backward-compatible.

- **Planner**: duplicate-query guard (`SearchAction.is_duplicate`) + bounded fallback query
- **Search Tool**: per-instance result cache + degrade-to-vdb on web *exception*
- **Reranker Tool**: bounded `max_candidates` window + ≤1-doc skip
- **Evidence Judge**: `marginal_gain` + `should_stop` plateau helpers
- **Answer Generator**: appearance-ordered, de-duplicated citations
- **Reward**: `early_stop_bonus` term (zero-default; surfaced in `retriever_aware()`)
- **Loop**: opt-in `evidence_plateau_min_gain` → `early_stops` metric (off by default, byte-identical)

## Verification
- New unit tests per component + reward + loop; full `pytest tests/unit` green
- GRPO smoke test passes
- Existing preset reward totals unchanged (new weight = 0); default loop behavior unchanged

## Docs
- Spec: `docs/superpowers/specs/2026-06-25-agent-framework-optimization-design.md`
- Plan: `docs/superpowers/plans/2026-06-25-agent-framework-optimization-plan.md`
- Tasks: `docs/superpowers/plans/2026-06-25-agent-framework-optimization-tasks.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

- **Spec coverage:** Planner (T1), Search Tool (T2), Reranker Tool (T3), Evidence Judge (T4), Answer Generator (T5), reward `early_stop_bonus` (T6), opt-in loop metric (T7), full-suite gate + PR (T8). Every spec success criterion maps to a task. ✓
- **Backward-compat constraints:** new args optional (T1–T3), new reward weight 0-default with regression assertion (T6), loop metric opt-in with default-unchanged assertion (T7). ✓
- **Type consistency:** `SearchAction.is_duplicate` (T1) used nowhere downstream; `EvidenceJudge.should_stop(prev, curr, min_gain)` defined in T4 and consumed in T7 with matching signature; `early_stop_bonus`/`early_stops` names consistent across T6 and T7. ✓
- **Placeholders:** T6 and T7 tests carry an explicit "match the existing harness" note rather than a fabricated factory name, because those two test files use file-specific fixtures the implementer must copy verbatim; the *invariant* asserted is concrete. All implementation steps show complete code. ✓
