# Assist First-Response Latency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the Assist request path freezing the event loop before retrieval, measure what "time to first response" actually costs, and cut the round's retrieval to one request.

**Architecture:** Four independent changes. Query enhancement moves off the event loop and its three LLM calls overlap. The streaming answer loop — which already iterates the provider's deltas — records time-to-first-token and time-to-first-claim and hands them up on the existing control-flow trace. The web app's lifespan pre-imports two lazily-imported modules. The AgenticRAG round replaces N single-query retrievals with one batched multi-query call.

**Tech Stack:** Python 3.12, asyncio, FastAPI, pytest + pytest-asyncio (`asyncio_mode=auto`), ruff.

**Spec:** `docs/superpowers/specs/2026-08-30-assist-first-response-latency-design.md`

## Global Constraints

- Never add a `torch` import to `src/context/`, `src/agents/`, or `src/internal/tools/` — CI runs a job with torch blocked from `sys.meta_path`, and these modules must import without it.
- `QueryEnhancer.enhance()` (sync) must keep working unchanged: `src/context/query_transform.py:98` constructs a `QueryEnhancer` from synchronous code.
- Access filters must be *enforced*, never merely forwarded — every retrieval call site that passes `filters` must also re-apply `filters.matches(...)` to the rows it gets back (invariant from #487–#492).
- Run `ruff check . --fix && ruff format .` before every commit; the repo's pre-commit hook aborts a commit on unformatted files.
- Every new test must be mutation-checked: revert the change under test, confirm that test turns red and the rest of the suite stays green, then restore.
- Full suite must stay green: `python3 -m pytest -q` (3703 tests on `2c133af`).

---

### Task 1: Un-block and parallelize query enhancement

**Files:**
- Modify: `tests/unit/test_no_blocking_calls_in_async.py` (the repo's AST guard)
- Modify: `src/context/query_enhancer.py` (add `enhance_async` after `enhance`, ~line 183)
- Modify: `src/agents/search/agentic_rag.py:223`
- Test: `tests/unit/test_no_blocking_calls_in_async.py`, `tests/unit/test_query_enhancer.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `QueryEnhancer.enhance_async(self, query: str) -> QueryBundle` — an async coroutine returning the same `QueryBundle` the sync `enhance` returns.

**Why the existing guard missed this.** `tests/unit/test_no_blocking_calls_in_async.py`
already walks every coroutine in `src/` looking for blocking calls, and its own
docstring names the lesson: "the fix was applied per-call-site by hand and
nothing checked the rest of the tree". It missed `enhance()` for a structural
reason — `BlockingCallFinder.visit_Call` matches `BLOCKING_FUNCTIONS` against
`getattr(func, "id", None)`, which is set only for **bare-name** calls like
`generate_answer(...)`. `self._enhancer.enhance(question)` is an attribute call,
where `func.id` does not exist and `func.attr` is `"enhance"`. So the guard can
see `generate_answer(...)` and can never see any method call at all.

Extending the finder to match method names is therefore the real fix, and it
makes the whole method-call bug class visible instead of this one instance.
`.enhance(` appears at exactly two call sites in `src/`
(`agents/search/agentic_rag.py:223`, inside a coroutine, and
`context/query_transform.py:98`, which is synchronous), so the new rule has no
false positives today.

- [ ] **Step 1: Extend the guard to see method calls**

In `tests/unit/test_no_blocking_calls_in_async.py`, add a new set beside
`BLOCKING_FUNCTIONS`:

```python
# Synchronous *methods* in this repo that perform blocking LLM/network IO.
# Matched on the attribute name, because the call site is `obj.method(...)`
# rather than a bare name. `enhance` runs three blocking LLM completions.
BLOCKING_METHODS = {"enhance"}
```

and add a branch to `BlockingCallFinder.visit_Call`, after the
`elif name in BLOCKING_FUNCTIONS:` branch:

```python
            elif attribute in BLOCKING_METHODS:
                self.findings.append((node.lineno, enclosing, f"{attribute}()"))
```

Add a finder unit test beside the existing finder tests:

```python
def test_the_finder_reports_a_blocking_method_inside_a_coroutine() -> None:
    source = """
async def prepare(self):
    return self._enhancer.enhance("q")
"""
    assert find_blocking_calls(source) == [(3, "prepare", "enhance()")]


def test_the_finder_ignores_a_blocking_method_offloaded_to_a_thread() -> None:
    source = """
import asyncio
async def prepare(self):
    return await asyncio.to_thread(self._enhancer.enhance, "q")
"""
    assert find_blocking_calls(source) == []
```

- [ ] **Step 2: Run the guard to verify it now fails**

Run: `python3 -m pytest tests/unit/test_no_blocking_calls_in_async.py -v`

Expected: `test_no_async_function_makes_a_blocking_call_directly` FAILS with
`src/agents/search/agentic_rag.py:223 async run() calls enhance()`. The two new
finder tests PASS. If the guard reports any *other* offender, stop and report
it — that is a second instance of this bug, not a false positive to suppress.

- [ ] **Step 3: Write the failing concurrency test**

Append to `tests/unit/test_query_enhancer.py`:

```python
async def test_enhance_async_matches_enhance():
    """The async path returns exactly what the sync path returns."""
    from src.context.query_enhancer import QueryEnhancer

    class _LLM:
        def complete(self, messages, **kw):
            return "sub one\nsub two"

    enhancer = QueryEnhancer(_LLM())
    assert await enhancer.enhance_async("q") == enhancer.enhance("q")


async def test_enhance_async_runs_strategies_concurrently():
    """decompose, hyde and step_back overlap instead of queueing.

    Guards the parallelism: run one after another they cost three LLM round
    trips of latency where they need only cost one.
    """
    import time

    from src.context.query_enhancer import QueryEnhancer

    in_flight = 0
    peak = 0

    class _SlowLLM:
        def complete(self, messages, **kw):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            time.sleep(0.05)
            in_flight -= 1
            return "sub one"

    await QueryEnhancer(_SlowLLM()).enhance_async("q")
    assert peak > 1, f"enhancement strategies were serialized (peak {peak})"
```

- [ ] **Step 4: Run it to verify it fails**

Run: `python3 -m pytest tests/unit/test_query_enhancer.py -v`
Expected: FAIL — `AttributeError: 'QueryEnhancer' object has no attribute 'enhance_async'`.

- [ ] **Step 5: Add `enhance_async` to `QueryEnhancer`**

Add `import asyncio` to the imports at the top of `src/context/query_enhancer.py`,
then add this method directly after `enhance`:

```python
    async def enhance_async(self, query: str) -> QueryBundle:
        """Async `enhance`: the three strategies run off-loop and concurrently.

        `llm.complete` is a blocking `requests` call, so calling `enhance` from
        a coroutine froze the event loop for all three round trips -- every
        other in-flight session stalled behind one user's query preparation.
        The strategies are independent and each already falls back on its own
        exception, so nothing here can raise and the round costs one round trip
        rather than three.
        """
        sub_queries, hyde_text, step_back_query = await asyncio.gather(
            asyncio.to_thread(self.decompose, query),
            asyncio.to_thread(self.hyde, query),
            asyncio.to_thread(self.step_back, query),
        )
        return QueryBundle(
            original=query,
            sub_queries=sub_queries,
            hyde_text=hyde_text,
            step_back_query=step_back_query,
        )
```

The sync `enhance` is left exactly as it is — `src/context/query_transform.py:98`
constructs a `QueryEnhancer` from synchronous code and still calls it.

- [ ] **Step 6: Await it from the agent loop**

In `src/agents/search/agentic_rag.py`, replace line 223:

```python
        bundle = self._enhancer.enhance(question)
```

with:

```python
        bundle = await self._enhancer.enhance_async(question)
```

- [ ] **Step 7: Run both test files to verify they pass**

Run: `python3 -m pytest tests/unit/test_no_blocking_calls_in_async.py tests/unit/test_query_enhancer.py tests/unit/test_agentic_rag.py -q`
Expected: PASS.

- [ ] **Step 8: Mutation-check both guards**

Revert `agentic_rag.py:223` to `self._enhancer.enhance(question)` and re-run.
Expected: `test_no_async_function_makes_a_blocking_call_directly` FAILS naming
that exact line. Restore. Then make `enhance_async` await its three
`asyncio.to_thread` calls one at a time instead of gathering them, and re-run.
Expected: `test_enhance_async_runs_strategies_concurrently` FAILS. Restore.

- [ ] **Step 9: Commit**

```bash
ruff check . --fix && ruff format .
git add tests/unit/test_no_blocking_calls_in_async.py tests/unit/test_query_enhancer.py src/context/query_enhancer.py src/agents/search/agentic_rag.py
git commit -m "perf(assist): run query enhancement off the event loop, concurrently"
```

---

### Task 2: Measure time-to-first-token and time-to-first-claim

**Files:**
- Modify: `src/context/models.py` (add `GenerationTimings` near `AnswerGenerationResult:305`, plus one field on `AnswerGenerationResult`)
- Modify: `src/context/pipeline.py:118-128` (call site) and `_generate_guarded_answer:159-360`
- Modify: `src/agents/core/control_flow_trace.py:17` (`ALLOWED_DETAIL_KEYS`)
- Modify: `src/agents/search/agentic_rag.py` (the `answer_generator` `_emit` after synthesis)
- Test: `tests/unit/test_generation_timings.py` (new), `tests/unit/test_agentic_rag.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `GenerationTimings` — frozen dataclass, fields `llm_first_token_ms: float | None = None`, `time_to_first_claim_ms: float | None = None`.
  - `AnswerGenerationResult.timings: GenerationTimings | None = None`.
  - `_generate_guarded_answer` returns a **9**-tuple: the existing 8 elements, then `GenerationTimings`.
  - The `answer_generator` / `synthesize` control-flow event gains `llm_first_token_ms` and `time_to_first_claim_ms` in its `details`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_generation_timings.py`:

```python
"""Time-to-first-token and time-to-first-claim on the grounded answer path."""

from __future__ import annotations

import time

from src.context.models import (
    AnswerGenerationRequest,
    ContextDocument,
    SearchContextBundle,
)
from src.context.pipeline import generate_answer

_DRAFT = (
    '{"abstain": false, "missing_information": [], "claims": '
    '[{"text": "FAISS is a vector search library", "evidence_ids": ["D1"]}]}'
)


class _StreamingLLM:
    """Streams the draft in pieces, slowly enough to time."""

    structured_output_capability = "prompt_only"

    def complete(self, messages, **kw):
        return _DRAFT

    def stream_complete(self, messages, **kw):
        time.sleep(0.02)          # provider think time before the first token
        for i in range(0, len(_DRAFT), 40):
            yield _DRAFT[i : i + 40]


def _request() -> AnswerGenerationRequest:
    context = SearchContextBundle(
        query="what is FAISS",
        documents=[
            ContextDocument(
                id="D1",
                title="FAISS",
                content="FAISS is a vector search library for dense retrieval.",
                url=None,
                score=1.0,
                metadata={},
            )
        ],
    )
    return AnswerGenerationRequest(question="what is FAISS", context=context)


def test_streaming_answer_reports_first_token_and_first_claim():
    claims: list[str] = []
    result = generate_answer(
        _request(), llm=_StreamingLLM(), on_claim=claims.append
    )

    assert result.timings is not None
    assert result.timings.llm_first_token_ms is not None
    assert result.timings.llm_first_token_ms >= 20
    if claims:
        assert result.timings.time_to_first_claim_ms is not None
        assert (
            result.timings.time_to_first_claim_ms
            >= result.timings.llm_first_token_ms
        )


def test_non_streaming_answer_reports_no_first_token():
    class _PlainLLM:
        structured_output_capability = "prompt_only"

        def complete(self, messages, **kw):
            return _DRAFT

    result = generate_answer(_request(), llm=_PlainLLM())
    assert result.timings is not None
    assert result.timings.llm_first_token_ms is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/unit/test_generation_timings.py -v`
Expected: FAIL — `AttributeError: 'AnswerGenerationResult' object has no attribute 'timings'`.

- [ ] **Step 3: Add the dataclass and the result field**

In `src/context/models.py`, add directly above `class AnswerGenerationResult`:

```python
@dataclass(frozen=True)
class GenerationTimings:
    """First-response latencies for one grounded generation.

    Both are None when they did not happen: no first token when the provider
    does not stream, no first claim when none was ever committed.
    """

    llm_first_token_ms: float | None = None
    time_to_first_claim_ms: float | None = None
```

and add this field to `AnswerGenerationResult` (after `structured_output_category`):

```python
    timings: "GenerationTimings | None" = None
```

- [ ] **Step 4: Record the timings in the streaming loop**

In `src/context/pipeline.py`, add `import time` beside the existing `import asyncio` (line 5) — the module does not import it today — and add
`from .models import GenerationTimings` to the one-import-per-line `.models`
block that starts at line 8.

Inside `_generate_guarded_answer`, before the `for attempt in range(max_attempts):` loop, add:

```python
    first_token_ms: float | None = None
    first_claim_ms: float | None = None
    t_gen = time.perf_counter()
```

In `_commit`, record the first claim — add as the last line of the function body:

```python
        nonlocal first_claim_ms
        if first_claim_ms is None:
            first_claim_ms = (time.perf_counter() - t_gen) * 1000
```

(`nonlocal` must be the first statement in `_commit`; put it at the top of the
function and the assignment at the bottom, after `on_claim(render_claim(claim))`.)

In the streaming branch, record the first delta — replace:

```python
                for delta in stream_fn(
                    active_prompt.messages,
                    **({"structured_output": schema_request} if schema_request else {}),
                ):
                    parts.append(delta)
```

with:

```python
                for delta in stream_fn(
                    active_prompt.messages,
                    **({"structured_output": schema_request} if schema_request else {}),
                ):
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - t_gen) * 1000
                    parts.append(delta)
```

- [ ] **Step 5: Return the timings as the 9th element**

Add a local helper inside `_generate_guarded_answer`, next to `_committed_answer`:

```python
    def _timings() -> GenerationTimings:
        return GenerationTimings(
            llm_first_token_ms=first_token_ms,
            time_to_first_claim_ms=first_claim_ms,
        )
```

Append `_timings()` as the last element of **every** `return` tuple in
`_generate_guarded_answer` — there are five: the two inside `_committed_answer`
and the three early-exit tuples (timeout, refused, and the trailing
`result is None` abstention), plus the function's final return. Grep for
`downgraded,` to find them all; each must gain one trailing `_timings(),`.

Then update the single call site at `src/context/pipeline.py:119-128`:

```python
        (
            answer,
            confidence,
            verification_status,
            retry_count,
            structured_output_requested,
            structured_output_applied,
            structured_output_downgraded,
            structured_output_category,
            generation_timings,
        ) = _generate_guarded_answer(request, llm, prompt, evidence, on_claim)
```

Initialize `generation_timings = None` alongside the other defaults near the top
of `generate_answer` (beside `retry_count = 0`), so the non-guarded branches
still have a value, and pass `timings=generation_timings` when constructing the
returned `AnswerGenerationResult`.

- [ ] **Step 6: Run the test to verify it passes**

Run: `python3 -m pytest tests/unit/test_generation_timings.py -v`
Expected: PASS.

- [ ] **Step 7: Allow the new detail keys through the trace sanitizer**

`ControlFlowRecorder` runs every `details` mapping through `_sanitize_details`,
which **drops any key not in `ALLOWED_DETAIL_KEYS`**. Without this step the two
new fields are silently discarded and Step 8's test fails for a reason that
looks nothing like the cause.

In `src/agents/core/control_flow_trace.py`, add both keys to the
`ALLOWED_DETAIL_KEYS` frozenset (line 17):

```python
        "decision",
        "llm_first_token_ms",
        "time_to_first_claim_ms",
```

Floats and `None` already pass the sanitizer's value check, so no other change
is needed there.

- [ ] **Step 8: Put the timings on the control-flow trace**

In `src/agents/search/agentic_rag.py`, the `_emit("answer_generator", "synthesize", ...)` call after synthesis gains two details:

```python
        _emit(
            "answer_generator",
            "synthesize",
            "completed",
            duration_ms=round((time.perf_counter() - t_syn) * 1000),
            citation_count=len(gen_result.citations),
            document_count=len(merged.documents),
            llm_first_token_ms=(
                gen_result.timings.llm_first_token_ms
                if gen_result.timings is not None
                else None
            ),
            time_to_first_claim_ms=(
                gen_result.timings.time_to_first_claim_ms
                if gen_result.timings is not None
                else None
            ),
        )
```

- [ ] **Step 9: Test the trace carries them**

`ControlFlowRecorder` takes `request_id` positionally, and recorded events are
read with `snapshot()` — there is no `.events` attribute.

Append to `tests/unit/test_agentic_rag.py`:

```python
@pytest.mark.asyncio
async def test_synthesis_trace_carries_first_response_timings():
    """The answer_generator trace event reports both first-response latencies."""
    import dataclasses

    from src.agents.core.control_flow_trace import ControlFlowRecorder
    from src.context.models import GenerationTimings

    def _timed_answer(request, **kwargs):
        return dataclasses.replace(
            _stub_generation_result(),
            timings=GenerationTimings(
                llm_first_token_ms=12.5, time_to_first_claim_ms=88.0
            ),
        )

    async def _retrieve(query, **kwargs):
        return _make_bundle(["d1"], query=query)

    recorder = ControlFlowRecorder("req-timings")
    llm = _llm_responses("sub", "hyde", "broader")
    with (
        patch("src.agents.search.agentic_rag.retrieve_context", _retrieve),
        patch("src.agents.search.agentic_rag.generate_answer", _timed_answer),
    ):
        loop = AgenticRAGLoop(AgenticRAGConfig(max_rounds=1), llm=llm)
        await loop.run("q", recorder=recorder)

    synth = [e for e in recorder.snapshot() if e.component == "answer_generator"]
    assert synth, "no answer_generator event recorded"
    assert synth[-1].details["llm_first_token_ms"] == 12.5
    assert synth[-1].details["time_to_first_claim_ms"] == 88.0
```

Note: if Task 4 has already landed, this test patches `retrieve_contexts`
instead — see Task 4 Step 7 for the `_batched` adapter.

- [ ] **Step 10: Run the tests**

Run: `python3 -m pytest tests/unit/test_generation_timings.py tests/unit/test_agentic_rag.py tests/unit/test_context_pipeline.py -q`
Expected: PASS. If a pipeline test unpacks `_generate_guarded_answer` directly, update it for the 9th element.

- [ ] **Step 11: Mutation-check**

Temporarily delete the `if first_token_ms is None:` assignment in the streaming
loop and re-run. Expected: `test_streaming_answer_reports_first_token_and_first_claim` FAILS. Restore.

- [ ] **Step 12: Commit**

```bash
ruff check . --fix && ruff format .
git add src/context/models.py src/context/pipeline.py src/agents/core/control_flow_trace.py src/agents/search/agentic_rag.py tests/unit/test_generation_timings.py tests/unit/test_agentic_rag.py
git commit -m "feat(observability): report time-to-first-token and time-to-first-claim"
```

---

### Task 3: Warm lazy imports at startup — WITHDRAWN

**Outcome:** implemented (8b9ef5f), proved inert by its own mutation check, and
reverted (48e972f). `app.py:69` already imports `request_capture` at top level
and pulls `src.context.safety` in transitively, so both modules are in
`sys.modules` before `_warm_lazy_imports` could run. Gutting the function to a
no-op left both of its tests passing — a false green. The steps below are kept
as the record of what was tried.

#### Original task text

**Files:**
- Modify: `src/internal/servers/web/app.py` (inside `lifespan`, which starts at line 1276)
- Test: `tests/unit/servers/web/test_import_warmup.py` (new)

**Interfaces:**
- Consumes: nothing.
- Produces: `_warm_lazy_imports() -> None` — module-level function in `app.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/servers/web/test_import_warmup.py`:

```python
"""The first request must not pay for imports the app can make at startup."""

from __future__ import annotations

import sys


def test_warm_lazy_imports_loads_the_deferred_modules():
    """Both modules deferred on the request path are importable up front.

    They are imported lazily inside hot functions -- `.safety` for laziness,
    `request_capture` to break a cycle from the agent loops. Neither cycle
    exists from the web app's own startup, and leaving them cold made the
    first request of a process pay ~1.4s of import time.
    """
    from src.internal.servers.web.app import _warm_lazy_imports

    for name in ("src.context.safety", "src.internal.servers.web.request_capture"):
        sys.modules.pop(name, None)

    _warm_lazy_imports()

    assert "src.context.safety" in sys.modules
    assert "src.internal.servers.web.request_capture" in sys.modules


def test_warm_lazy_imports_never_raises(monkeypatch):
    """A failed warm-up must not take down startup."""
    import builtins

    from src.internal.servers.web.app import _warm_lazy_imports

    real_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if name == "src.context.safety":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)
    _warm_lazy_imports()  # must not raise
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/unit/servers/web/test_import_warmup.py -v`
Expected: FAIL — `ImportError: cannot import name '_warm_lazy_imports'`.

- [ ] **Step 3: Implement the warm-up**

Add at module level in `src/internal/servers/web/app.py`, near the other
module-level helpers:

```python
def _warm_lazy_imports() -> None:
    """Import modules the request path imports lazily.

    `src.context.safety` and `request_capture` are imported inside hot
    functions -- the first for laziness, the second to break an import cycle
    from the agent-loop modules. That cycle does not exist from the web app's
    own startup, where the web package is already fully loaded. Left cold, the
    first request of a process paid roughly 1.4s of import time, charged to
    whichever user happened to arrive first.

    Best-effort: a warm-up failure must never stop the process from starting.
    """
    for module in ("src.context.safety", "src.internal.servers.web.request_capture"):
        try:
            import_module(module)
        except Exception:  # noqa: BLE001 -- warming is best-effort
            logger.warning("Could not warm %s", module, exc_info=True)
```

Add `from importlib import import_module` to the imports at the top of `app.py`
if it is not already there.

- [ ] **Step 4: Call it from lifespan**

In `lifespan` (line ~1276), add `_warm_lazy_imports()` immediately after
`seed_db(db)`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/unit/servers/web/test_import_warmup.py -v`
Expected: PASS.

- [ ] **Step 6: Mutation-check**

Temporarily make `_warm_lazy_imports` a no-op (`return` as the first statement)
and re-run. Expected: `test_warm_lazy_imports_loads_the_deferred_modules`
FAILS. Restore.

- [ ] **Step 7: Commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/servers/web/app.py tests/unit/servers/web/test_import_warmup.py
git commit -m "perf(web): warm request-path lazy imports during startup"
```

---

### Task 4: Batch the AgenticRAG round's retrievals

**Files:**
- Modify: `src/context/pipeline.py` (add `retrieve_contexts` after `retrieve_context:40-50`)
- Modify: `src/context/retrieval/search_runner.py` (add `build_search_contexts` after `build_search_context:74-93`)
- Modify: `src/agents/search/agentic_rag.py` (the round's retrieval fan-out, ~line 252)
- Test: `tests/unit/test_search_runner.py`, `tests/unit/test_agentic_rag.py`

**Interfaces:**
- Consumes: nothing from Tasks 1–3.
- Produces:
  - `build_search_contexts(queries: list[str], *, top_k: int = 5, filters: SearchFilters | None = None, search_url: str = ..., timeout_seconds: int = 15, max_retries: int = 3) -> list[SearchContextBundle]` in `search_runner.py`.
  - `retrieve_contexts(questions: list[str], *, search_url: str = ..., top_k: int = 5, filters: SearchFilters | None = None) -> list[SearchContextBundle]` in `pipeline.py` — one bundle per input query, in input order.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_search_runner.py` — no test file references
`build_search_context` today, so this is a new file:

```python
async def test_build_search_contexts_issues_one_request_for_all_queries():
    """N queries cost one round trip, not N.

    Guards the batch: the retrieval API takes {"queries": [...]}, and issuing
    one single-query request per query paid N session setups.
    """
    from unittest.mock import patch

    from src.context.retrieval.search_runner import build_search_contexts
    from src.context.search import SearchResult

    calls: list[list[str]] = []

    async def _fake_retrieve(self, queries, topk=None, filters=None):
        calls.append(list(queries))
        return [
            [SearchResult(contents=f"about {q}", title=q, url=f"http://x/{q}")]
            for q in queries
        ]

    with patch(
        "src.context.retrieval.client.SearchClient.retrieve", _fake_retrieve
    ):
        bundles = await build_search_contexts(
            ["alpha", "beta", "gamma"], top_k=5, search_url="http://x/retrieve"
        )

    assert len(calls) == 1, f"expected one batched request, got {len(calls)}"
    assert calls[0] == ["alpha", "beta", "gamma"]
    assert [b.query for b in bundles] == ["alpha", "beta", "gamma"]
    assert bundles[0].documents[0].content.endswith("alpha")


async def test_build_search_contexts_returns_empty_bundles_for_no_queries():
    from src.context.retrieval.search_runner import build_search_contexts

    assert await build_search_contexts([], search_url="http://x/retrieve") == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/unit/test_search_runner.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_search_contexts'`.

- [ ] **Step 3: Implement the batched runner**

Add to `src/context/retrieval/search_runner.py`, after `build_search_context`:

```python
async def build_search_contexts(
    queries: list[str],
    *,
    top_k: int = 5,
    filters: SearchFilters | None = None,
    search_url: str = "http://localhost:8000/retrieve",
    timeout_seconds: int = 15,
    max_retries: int = 3,
) -> list[SearchContextBundle]:
    """Retrieve for several queries in one request; one bundle per query.

    The retrieval API is natively multi-query ({"queries": [...]}), so N
    independent queries cost one round trip on one session rather than N of
    each. Bundles come back in input order and are built exactly as
    `build_search_context` builds a single one, so this is a transport change
    only.

    Retrieval provider only: the multi-query request shape is specific to
    /retrieve.
    """
    if not queries:
        return []
    client = SearchClient(
        SearchClientConfig(
            url=search_url,
            topk=top_k,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    )
    try:
        rows = await client.retrieve(
            queries,
            topk=top_k,
            filters=filters.to_payload() if filters is not None else None,
        )
    finally:
        await client.aclose()
    # Enforce, don't just forward: a third-party backend need not honour the
    # forwarded filter, and anything returned here reaches a model's context.
    return [
        build_context_bundle(query, _apply_filters(row, filters), max_documents=top_k)
        for query, row in zip(queries, rows)
    ]
```

Add `SearchContextBundle` and `SearchFilters` to the module's imports from
`..models` if they are not already imported there.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/unit/test_search_runner.py -v`
Expected: PASS.

- [ ] **Step 5: Add the pipeline wrapper**

Add to `src/context/pipeline.py`, directly after `retrieve_context`:

```python
async def retrieve_contexts(
    questions: list[str],
    *,
    search_url: str = "http://localhost:8000/retrieve",
    top_k: int = 5,
    filters: SearchFilters | None = None,
) -> list[SearchContextBundle]:
    """Batched `retrieve_context`: one bundle per question, in input order."""
    return await build_search_contexts(
        questions, top_k=top_k, filters=filters, search_url=search_url
    )
```

Import `build_search_contexts` next to the existing `build_search_context`
import in that module.

- [ ] **Step 6: Use it in the agent loop**

In `src/agents/search/agentic_rag.py`, replace the `asyncio.gather` fan-out
(added in #560) with one batched call. Replace:

```python
            contexts = await asyncio.gather(
                *(
                    retrieve_context(
                        q,
                        search_url=self.config.retrieval_url,
                        top_k=self.config.topk,
                        filters=self.config.filters,
                    )
                    for q in novel_queries
                ),
                return_exceptions=True,
            )
```

with:

```python
            # One request for the whole round: the retrieval API is natively
            # multi-query, so N queries cost one round trip on one session
            # instead of N concurrent ones. A transport failure fails the
            # round's queries together, which the per-query handler below
            # reports exactly as it reported an individual failure.
            try:
                contexts: list[object] = list(
                    await retrieve_contexts(
                        novel_queries,
                        search_url=self.config.retrieval_url,
                        top_k=self.config.topk,
                        filters=self.config.filters,
                    )
                )
            except Exception as exc:  # noqa: BLE001 -- degrade, as before
                contexts = [exc] * len(novel_queries)
```

Change the import at the top of the module from `retrieve_context` to
`retrieve_context, retrieve_contexts` — `retrieve_context` is still referenced
by tests that patch it, so keep it imported. The `for q, ctx in
zip(novel_queries, contexts):` loop below is unchanged.

- [ ] **Step 7: Update the AgenticRAG tests that patch `retrieve_context`**

The existing tests patch `src.agents.search.agentic_rag.retrieve_context`. Repoint
them at `retrieve_contexts`, which takes a list and returns a list. Add a helper
near `_make_bundle` in `tests/unit/test_agentic_rag.py`:

```python
def _batched(per_query):
    """Adapt a single-query fake into the batched retrieve_contexts shape."""

    async def _call(queries, **kwargs):
        return [await per_query(q, **kwargs) for q in queries]

    return _call
```

Repoint each `patch("src.agents.search.agentic_rag.retrieve_context", X)` to
`patch("src.agents.search.agentic_rag.retrieve_contexts", _batched(X))`, where
`X` is that test's existing single-query fake. For tests using
`AsyncMock(return_value=bundle)`, pass an equivalent async function instead.
The concurrency guard `test_round_retrievals_run_concurrently` from #560 no
longer applies — a single batched request has no fan-out to overlap — so delete
it and note the replacement in the commit message.

- [ ] **Step 8: Run the full retrieval and loop tests**

Run: `python3 -m pytest tests/unit/test_search_runner.py tests/unit/test_agentic_rag.py tests/unit/test_search_filters_plumbing.py tests/unit/test_search_route_access_filters.py -q`
Expected: PASS.

- [ ] **Step 9: Mutation-check**

Temporarily change `build_search_contexts` to loop `build_search_context` per
query and re-run. Expected:
`test_build_search_contexts_issues_one_request_for_all_queries` FAILS on the
`len(calls) == 1` assertion. Restore.

- [ ] **Step 10: Commit**

```bash
ruff check . --fix && ruff format .
git add src/context/pipeline.py src/context/retrieval/search_runner.py src/agents/search/agentic_rag.py tests/unit/test_search_runner.py tests/unit/test_agentic_rag.py
git commit -m "perf(assist): batch a retrieval round into one multi-query request"
```

---

### Task 5: Verify the whole branch

**Files:** none modified unless a check fails.

- [ ] **Step 1: Lint**

Run: `ruff check . && ruff format --check .`
Expected: clean.

- [ ] **Step 2: Full suite**

Run: `python3 -m pytest -q`
Expected: all pass, count at or above 3703 plus the tests added here.

- [ ] **Step 3: Torch-free import check**

The CI unit-test job runs with `torch` unavailable. Confirm the changed modules
still import:

```bash
python3 - <<'PY'
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"blocked: {name}")
        return None
sys.meta_path.insert(0, Block())
import src.context.pipeline, src.context.query_enhancer, src.context.models
import src.context.retrieval.search_runner, src.agents.search.agentic_rag
print("torch-free import OK")
PY
```

Expected: `torch-free import OK`.

- [ ] **Step 4: Re-measure**

Re-run the latency probes from the spec against a live demo retrieval server on
the scifact corpus, and record before/after numbers in the PR body:

```bash
python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus_scifact.jsonl &
```

Expected: the round's retrieval drops from ~13.5 ms to ~6.3 ms, and the
event-loop stall during enhancement drops from ~320 ms to ~0 ms.

- [ ] **Step 5: Push and open the PR**

```bash
git push -u origin perf/assist-first-response-latency
```

Open the PR against `main` with a body covering: the measured before/after
table, the four changes, what was deliberately left out (TTS, the dead
`EvalTimings` field, the unwired latency middleware), and links to the spec and
this plan.

---

## Notes for the executor

- `src/context/query_transform.py:98` builds a `QueryEnhancer` synchronously. Task 1 must not change or remove the sync `enhance`.
- Task 4 deletes a test added in #560. That is intentional and is called out in Task 4 Step 7 — the behaviour it guarded (a round not costing N sequential round trips) is guarded better by the new one-request assertion.
- Tasks 1, 2 and 3 are independent of each other and of Task 4. If Task 4 turns out to break access-filter enforcement, drop it and ship the rest; the spec records this as an accepted outcome.
