# Agent Framework Cost Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce low-value Agent Framework search/rerank work while preserving evidence safety rails and citation behavior.

**Architecture:** Keep the existing `SearchAgentLoop` architecture and add small cost-aware guards at the current action execution points. Rerank gating stays in `_execute_search_round` before results are added to `AgentContext`, and normalized repeat detection stays in `_partition_search_requests` before retrieval is dispatched.

**Tech Stack:** Python 3.11+, async `SearchAgentLoop`, dataclass-based search results, pytest unit tests, existing GRPO reward metrics.

## Global Constraints

- Optimize latency/cost first, with only small architecture hardening where it directly supports cost-saving behavior.
- Do not add a model head, dependency, retriever, or reranker.
- Do not change evidence sufficiency gating, answer rejection behavior, citation formatting, or Answer Generator behavior.
- Keep reranking opt-in via the existing per-search `rerank="true"` flag.
- Preserve existing reward cost semantics: `rerank_calls` counts search rounds where reranking actually ran.
- Keep skipped rerank metrics observational by default.

---

## File Structure

- Modify `src/agents/search.py`: add normalized query helper, normalized repeat filtering, rerank requested/skipped metrics, and conservative rerank gating.
- Modify `tests/unit/test_agent_loop.py`: add focused tests for skipped reranks and normalized repeated queries.
- Keep `src/agents/components/*` unchanged; this pass does not move cost policy into component classes.
- Keep `src/training/reward.py` unchanged; reward already reads `rerank_calls`.

### Task 1: Rerank Request Metrics And Cost Gating

**Files:**
- Modify: `src/agents/search.py`
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: existing `_execute_search_round(..., metrics: dict[str, float], rerank: bool) -> SearchRoundResult`
- Produces: metrics `rerank_requested`, `rerank_calls`, `rerank_skipped`; helper `_should_rerank(results_by_query: list[list[SearchResult]]) -> bool`

- [x] **Step 1: Write failing tests for skipped reranks**

Add these tests near the existing rerank tests in `tests/unit/test_agent_loop.py`:

```python
def test_search_agent_loop_skips_rerank_for_single_result_round():
    """A one-document round cannot benefit from rerank, so the reranker is not called."""
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager(
        [
            tokenizer.encode('<search rerank="true">q</search>'),
            tokenizer.encode("<answer>a [R1Q1D1]</answer>"),
        ]
    )
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {("q",): [[SearchResult(contents="single body content", score=0.5)]]}
    )
    calls: list[str] = []

    def reranker(query: str, docs: list[SearchResult]) -> list[SearchResult]:
        calls.append(query)
        return docs

    loop._reranker = reranker

    output = asyncio.run(
        loop.run([{"role": "user", "content": "q?"}], {"temperature": 0.0})
    )

    assert calls == []
    assert output.metrics["rerank_requested"] == 1.0
    assert output.metrics["rerank_calls"] == 0.0
    assert output.metrics["rerank_skipped"] == 1.0
```

Add a second test for empty results:

```python
def test_search_agent_loop_counts_empty_rerank_request_as_skipped():
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager(
        [
            tokenizer.encode('<search rerank="true">q</search>'),
            tokenizer.encode("<answer>a</answer>"),
        ]
    )
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        search_config=SearchAgentLoopConfig(
            max_turns=3,
            require_sufficient_evidence_before_answer=False,
        ),
    )
    loop._search_client = FakeSearchClient({("q",): [[]]})
    calls: list[str] = []
    loop._reranker = lambda query, docs: calls.append(query) or docs

    output = asyncio.run(
        loop.run([{"role": "user", "content": "q?"}], {"temperature": 0.0})
    )

    assert calls == []
    assert output.metrics["rerank_requested"] == 1.0
    assert output.metrics["rerank_calls"] == 0.0
    assert output.metrics["rerank_skipped"] == 1.0
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_agent_loop.py -k "rerank" -v`

Expected: the new tests fail because `rerank_requested` and `rerank_skipped` are missing, and the single-result reranker is still called.

- [x] **Step 3: Add metrics and gating**

In `src/agents/search.py`, add metric defaults in the metrics dictionary created in `SearchAgentLoop.run`:

```python
"rerank_requested": 0.0,
"rerank_calls": 0.0,
"rerank_skipped": 0.0,
```

Add this static helper near `_cache_key`:

```python
@staticmethod
def _should_rerank(results_by_query: list[list[SearchResult]]) -> bool:
    """Rerank only when at least one query has two or more results."""
    return any(len(results) >= 2 for results in results_by_query)
```

Replace the existing rerank block in `_execute_search_round` with:

```python
if rerank:
    metrics["rerank_requested"] += 1.0
    if self._reranker is not None and self._should_rerank(results_by_query):
        results_by_query = [
            self._reranker(query, results)
            for query, results in zip(queries, results_by_query)
        ]
        metrics["rerank_calls"] += 1.0
    else:
        metrics["rerank_skipped"] += 1.0
```

- [x] **Step 4: Run rerank tests**

Run: `pytest tests/unit/test_agent_loop.py -k "rerank" -v`

Expected: all rerank tests pass, including the existing test where two results are reordered and `rerank_calls == 1.0`.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/agents/search.py tests/unit/test_agent_loop.py docs/superpowers/plans/2026-06-25-agent-framework-cost-optimization.md
git commit -m "feat: gate low-value rerank requests"
```

### Task 2: Normalized Repeated Query Blocking

**Files:**
- Modify: `src/agents/search.py`
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: existing `_partition_search_requests(query_specs, executed_queries, rounds_used)`
- Produces: helper `_normalize_query_key(query: str) -> str`; `executed_queries` stores normalized query keys

- [ ] **Step 1: Write failing test for normalized repeats**

Add this test near `test_search_agent_loop_skips_repeated_queries_with_feedback` in `tests/unit/test_agent_loop.py`:

```python
def test_search_agent_loop_skips_repeated_queries_after_whitespace_normalization():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<search>alpha   query</search>"),
        tokenizer.encode("<search> alpha query </search>"),
        tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha   query",): [
                [
                    SearchResult(
                        contents='"Doc A"\nAlpha body', url="https://example.com/a"
                    )
                ],
            ],
        }
    )
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    third_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[2]["prompt_ids"]
    )
    assert fake_client.calls == [["alpha   query"]]
    assert "Repeated search skipped" in third_prompt
    assert output.metrics["repeated_search_queries"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_agent_loop.py -k "repeated_queries" -v`

Expected: the new test fails because the second query is currently treated as distinct.

- [ ] **Step 3: Add normalized query keys**

In `src/agents/search.py`, add this helper near `_normalize_task_id`:

```python
def _normalize_query_key(query: str) -> str:
    """Stable key for repeat detection without semantic rewriting."""
    return " ".join(query.strip().split())
```

Update `_partition_search_requests`:

```python
for task_id, query in query_specs:
    query_key = _normalize_query_key(query)
    if query_key in executed_queries:
        repeated.append(query)
    elif at_limit:
        overflow.append(query)
    else:
        allowed.append((task_id, query))
```

Update the call site after a search round is accepted:

```python
executed_queries.update(_normalize_query_key(q) for q in search_tool_call.queries)
```

- [ ] **Step 4: Run repeated-query tests**

Run: `pytest tests/unit/test_agent_loop.py -k "repeated_queries" -v`

Expected: both exact repeated query and normalized whitespace repeated query tests pass.

- [ ] **Step 5: Run focused regression**

Run:

```bash
pytest tests/unit/test_agent_loop.py -k "rerank or repeated" -v
pytest tests/unit/test_components.py -v
pytest tests/unit/test_reward.py -k "rerank or retriever_aware" -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/agents/search.py tests/unit/test_agent_loop.py docs/superpowers/plans/2026-06-25-agent-framework-cost-optimization.md
git commit -m "feat: normalize repeated search queries"
```

## Final Verification

- [ ] Run `git diff --check`.
- [ ] Run `pytest tests/unit/test_agent_loop.py -k "rerank or repeated" -v`.
- [ ] Run `pytest tests/unit/test_components.py -v`.
- [ ] Run `pytest tests/unit/test_reward.py -k "rerank or retriever_aware" -v`.
- [ ] Run `git status --short` and confirm only intended files are changed or the branch is clean.
