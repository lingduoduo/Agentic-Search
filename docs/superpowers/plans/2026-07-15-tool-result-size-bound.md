# Bounded Tool Result Size Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject a registered tool's result when its serialized JSON exceeds a size cap, so that an oversized result cannot enter the RAG prompt. The prompt-size bound is the guarantee this delivers unconditionally; the incremental encoder also reduces — but does not cap — peak memory and event-loop time, because a result whose bulk is a single large scalar is encoded in full before the first size check.

**Architecture:** Replace the unbounded `json.dumps` in `collect_tool_evidence` with an incremental `JSONEncoder.iterencode` loop that raises `ValueError` as soon as the running length exceeds `max_result_chars`. The raise lands in the function's existing `except Exception`, which already marks the tool `"failed"` and continues — so oversized results degrade to retrieval-only answering with no new status, branch, or trace attribute. Thread a matching parameter through `answer_with_retrieval`.

**Tech Stack:** Python 3.11+, stdlib `json`, pytest + pytest-asyncio, ruff.

Spec: `docs/superpowers/specs/2026-07-15-tool-result-size-bound-design.md`

## Global Constraints

- Default cap is exactly `8192` characters, in both `collect_tool_evidence` and `answer_with_retrieval`.
- The boundary is exclusive: a serialized length of exactly `max_result_chars` is **accepted**; `max_result_chars + 1` is rejected. The check is `if total > max_result_chars`, never `>=`.
- **Never pass `_one_shot=True` to `iterencode`.** `JSONEncoder.iterencode(o)` defaults to `_one_shot=False`, selecting the pure-Python generator that yields many small chunks. `_one_shot=True` selects the C encoder, which returns the whole document as a single chunk — the size check would then fire only after the entire result was already encoded. The prompt-size bound would survive that (the single chunk is still counted before use), but the early-abort work saving would not: an over-cap result whose bulk is spread across many scalars would pay for full serialization instead of aborting near-instantly (~580x slower), while tests that only inspect returned evidence still pass.
- The encoder must keep the exact settings the current code uses: `sort_keys=True, separators=(",", ":"), ensure_ascii=False`. Output must stay byte-identical to today's `json.dumps` for under-cap results.
- Reject; never truncate. Truncated JSON is invalid JSON and can silently drop a negation, turning a result into evidence that misleads the lexical verifier.
- Do not add a new status string. Oversized results reuse the existing `"failed"` status.
- Follow the module's existing validation pattern: raise `ValueError` from the guard block at the top of `collect_tool_evidence`, before any selection or invocation.
- Run `ruff check` and `ruff format` on every modified file before committing.

---

### Task 1: Bound result encoding in `collect_tool_evidence`

**Files:**
- Modify: `src/context/tool_evidence.py` (signature ~58-66, guards 72-75, encode call 134-136; add module-level helper)
- Test: `tests/unit/test_rag_tool_evidence.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `collect_tool_evidence(query, registry, selector, *, max_calls: int = 2, timeout_seconds: float = 5.0, max_result_chars: int = 8192, status_callback: Callable[[str, str], None] | None = None) -> list[EvidenceSource]` — Task 2 calls this with `max_result_chars=`.
  - `_encode_bounded(result: object, max_result_chars: int) -> str` — module-private; raises `ValueError` if the encoding exceeds the cap. Task 1's own test imports it directly.

The existing test file already defines `Registry` and `Selector` stubs at the top (`tests/unit/test_rag_tool_evidence.py:15-41`). Reuse them; do not redefine them.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_rag_tool_evidence.py`:

```python
@pytest.mark.asyncio
async def test_oversized_tool_result_is_rejected_and_reported_failed():
    registry = Registry(
        [ToolDescriptor("bulk", "Bulk data", ToolSafety.READ_ONLY)],
        {"bulk": {"blob": "x" * 5000}},
    )
    selector = Selector([ToolRequest("bulk")])
    statuses: list[tuple[str, str]] = []

    evidence = await collect_tool_evidence(
        "query",
        registry,
        selector,
        max_result_chars=1000,
        status_callback=lambda name, status: statuses.append((name, status)),
    )

    assert evidence == []
    assert statuses == [("bulk", "failed")]


@pytest.mark.asyncio
async def test_result_at_exactly_the_cap_is_accepted_and_one_over_is_rejected():
    result = {"value": "x" * 10}
    exact = json.dumps(
        result, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    registry = Registry(
        [ToolDescriptor("probe", "Probe", ToolSafety.READ_ONLY)], {"probe": result}
    )
    selector = Selector([ToolRequest("probe")])

    accepted = await collect_tool_evidence(
        "query", registry, selector, max_result_chars=len(exact)
    )
    assert [item.text for item in accepted] == [exact]

    rejected = await collect_tool_evidence(
        "query", registry, selector, max_result_chars=len(exact) - 1
    )
    assert rejected == []


@pytest.mark.asyncio
async def test_under_cap_encoding_is_byte_identical_to_json_dumps():
    result = {"z": 1.5, "a": "café", "n": [1, 2, {"k": None}]}
    registry = Registry(
        [ToolDescriptor("probe", "Probe", ToolSafety.READ_ONLY)], {"probe": result}
    )
    selector = Selector([ToolRequest("probe")])

    evidence = await collect_tool_evidence("query", registry, selector)

    assert evidence[0].text == json.dumps(
        result, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def test_bounded_encoding_aborts_before_walking_the_whole_structure():
    # Observing early abort requires calling the encoder directly: through
    # collect_tool_evidence both outcomes collapse to the same "failed" status.
    # A sentinel json cannot serialize sits past the cap. Aborting early raises
    # ValueError without ever reaching it; encoding the whole structure would
    # instead raise TypeError. This test fails if _one_shot=True is introduced.
    class Unserializable:
        pass

    result = ["y" * 1000 for _ in range(50)] + [Unserializable()]

    with pytest.raises(ValueError, match="max_result_chars"):
        _encode_bounded(result, 8192)


@pytest.mark.asyncio
async def test_oversized_result_does_not_prevent_other_tool_evidence():
    registry = Registry(
        [
            ToolDescriptor("bulk", "Bulk data", ToolSafety.READ_ONLY),
            ToolDescriptor("small", "Small data", ToolSafety.READ_ONLY),
        ],
        {"bulk": {"blob": "x" * 5000}, "small": {"ok": True}},
    )
    selector = Selector([ToolRequest("bulk"), ToolRequest("small")])
    statuses: list[tuple[str, str]] = []

    evidence = await collect_tool_evidence(
        "query",
        registry,
        selector,
        max_result_chars=1000,
        status_callback=lambda name, status: statuses.append((name, status)),
    )

    assert [item.tool_name for item in evidence] == ["small"]
    assert evidence[0].id == "T1"
    assert statuses == [("bulk", "failed"), ("small", "succeeded")]
```

Add `_encode_bounded` to the existing import block at the top of the file
(`tests/unit/test_rag_tool_evidence.py:7-13`), which currently imports from `src.context`.
`_encode_bounded` is private and not re-exported, so import it from its module:

```python
from src.context.tool_evidence import _encode_bounded
```

Extend the existing `test_invalid_limits_reject_without_selecting_or_invoking`
(`tests/unit/test_rag_tool_evidence.py:280-287`) by adding these two assertions to its body:

```python
    with pytest.raises(ValueError, match="max_result_chars"):
        await collect_tool_evidence("query", registry, selector, max_result_chars=0)
    with pytest.raises(ValueError, match="max_result_chars"):
        await collect_tool_evidence("query", registry, selector, max_result_chars=-1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_rag_tool_evidence.py -q`

Expected: FAIL. The `_encode_bounded` import raises `ImportError: cannot import name '_encode_bounded'`, which errors collection of the whole module. That single error is the expected RED signal for every test above.

- [ ] **Step 3: Add the bounded encoder helper**

In `src/context/tool_evidence.py`, add this function at module level, directly above the existing `_freeze_mapping` helper (currently line 155):

```python
def _encode_bounded(result: object, max_result_chars: int) -> str:
    """Serialize ``result`` to canonical JSON, aborting once it exceeds the cap.

    ``iterencode`` must not be passed ``_one_shot=True``: that selects the C
    encoder, which returns the whole document as one chunk. The prompt-size
    bound would survive that (the single chunk is still counted before use),
    but the early-abort work saving would not: an over-cap result whose bulk
    is spread across many scalars would pay for full serialization instead of
    aborting near-instantly (~580x slower, by prior measurement).
    """
    encoder = json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    chunks: list[str] = []
    total = 0
    for chunk in encoder.iterencode(result):
        total += len(chunk)
        if total > max_result_chars:
            raise ValueError("tool result exceeds max_result_chars")
        chunks.append(chunk)
    return "".join(chunks)
```

- [ ] **Step 4: Add the parameter, its guard, and use the helper**

In `src/context/tool_evidence.py`, add `max_result_chars` to the signature (currently lines 58-66), keyword-only, after `timeout_seconds`:

```python
async def collect_tool_evidence(
    query: str,
    registry: ToolRegistry,
    selector: ToolSelector,
    *,
    max_calls: int = 2,
    timeout_seconds: float = 5.0,
    max_result_chars: int = 8192,
    status_callback: Callable[[str, str], None] | None = None,
) -> list[EvidenceSource]:
```

Add this guard immediately after the existing `timeout_seconds` guard (currently lines 74-75):

```python
    if max_result_chars <= 0:
        raise ValueError("max_result_chars must be positive")
```

Replace the `json.dumps` call (currently lines 134-136) with the helper. The surrounding
`try` / `except Exception` block stays exactly as it is:

```python
            result = await asyncio.wait_for(
                registry.invoke(request), timeout=timeout_seconds
            )
            text = _encode_bounded(result, max_result_chars)
```

Update the function's docstring (currently lines 67-71) to record the new rejection reason:

```python
    """Collect normalized evidence from explicitly read-only tools.

    Invalid selections, invocation failures, timeouts, results that exceed
    ``max_result_chars`` once serialized, and results that cannot be represented
    as JSON are ignored so retrieval-based answering can continue.
    """
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_rag_tool_evidence.py -q`

Expected: PASS, all tests in the file.

- [ ] **Step 6: Run the surrounding regression and lint**

Run: `python3 -m pytest tests/unit/test_rag_tool_evidence.py tests/unit/test_rag_pipeline_integration.py tests/unit/test_mcp_server.py -q`

Expected: PASS. `test_read_only_tools_supply_stable_normalized_evidence` (line 65) already asserts `evidence[0].text == json.dumps(...)`; it passing is the proof that the `iterencode` swap changed no existing output.

Run: `python3 -m ruff check src/context/tool_evidence.py tests/unit/test_rag_tool_evidence.py && python3 -m ruff format src/context/tool_evidence.py tests/unit/test_rag_tool_evidence.py`

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add src/context/tool_evidence.py tests/unit/test_rag_tool_evidence.py
git commit -m "feat: bound tool result size in collect_tool_evidence"
```

---

### Task 2: Thread the cap through `answer_with_retrieval`

**Files:**
- Modify: `src/context/pipeline.py` (signature ~276, `collect_tool_evidence` call ~297-306)
- Test: `tests/unit/test_rag_pipeline_integration.py`

**Interfaces:**
- Consumes: `collect_tool_evidence(..., max_result_chars: int = 8192, ...)` from Task 1.
- Produces: `answer_with_retrieval(..., max_tool_result_chars: int = 8192, ...)`.

That test file defines its own `Registry` / `Selector` stubs and the `_async` / `_context` helpers (see `tests/unit/test_rag_pipeline_integration.py:76-124`). Reuse them; do not redefine them.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_rag_pipeline_integration.py`:

```python
@pytest.mark.asyncio
async def test_pipeline_answers_from_retrieval_when_tool_result_is_oversized(
    monkeypatch,
):
    registry = Registry(
        [ToolDescriptor("bulk", safety=ToolSafety.READ_ONLY)],
        {"bulk": {"blob": "x" * 5000}},
    )
    selector = Selector([ToolRequest("bulk")])
    monkeypatch.setattr(
        "src.context.pipeline.retrieve_context", lambda *a, **k: _async(_context())
    )

    result = await answer_with_retrieval(
        "FAISS status",
        tool_registry=registry,
        tool_selector=selector,
        max_tool_result_chars=1000,
    )

    assert [request.tool_name for request in registry.calls] == ["bulk"]
    assert result.tool_evidence == []
    assert result.answer == "FAISS search is currently available. [D1]"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/unit/test_rag_pipeline_integration.py::test_pipeline_answers_from_retrieval_when_tool_result_is_oversized -q`

Expected: FAIL with `TypeError: answer_with_retrieval() got an unexpected keyword argument 'max_tool_result_chars'`.

- [ ] **Step 3: Add the parameter and pass it through**

In `src/context/pipeline.py`, add to the signature directly after `max_tool_calls: int = 2` (currently line 276):

```python
    max_tool_result_chars: int = 8192,
```

Pass it at the `collect_tool_evidence` call site (currently lines 297-306), after `timeout_seconds`:

```python
            tool_evidence = await collect_tool_evidence(
                question,
                tool_registry,
                tool_selector,
                max_calls=max_tool_calls,
                timeout_seconds=tool_timeout_seconds,
                max_result_chars=max_tool_result_chars,
                status_callback=lambda name, status: tool_statuses.append(
                    (name, status)
                ),
            )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/unit/test_rag_pipeline_integration.py -q`

Expected: PASS, all tests in the file.

- [ ] **Step 5: Run the full unit suite and lint**

Run: `env HF_HOME=/tmp/agentic-search-hf-cache python3 -m pytest -q tests/unit`

Expected: PASS with no failures. The pre-existing baseline on this branch is 2616 passed; expect that plus the tests added by Tasks 1 and 2.

Run: `python3 -m ruff check src/context/pipeline.py tests/unit/test_rag_pipeline_integration.py && python3 -m ruff format src/context/pipeline.py tests/unit/test_rag_pipeline_integration.py && git diff --check`

Expected: `All checks passed!` and `git diff --check` exits 0.

- [ ] **Step 6: Commit**

```bash
git add src/context/pipeline.py tests/unit/test_rag_pipeline_integration.py
git commit -m "feat: thread tool result size cap through answer_with_retrieval"
```

---

### Task 3: Document the bound

**Files:**
- Modify: `docs/retrieval.md` (the tool-evidence section that documents bounded read-only execution, near lines 96-116)

**Interfaces:**
- Consumes: the parameter names from Tasks 1 and 2 — `max_result_chars`, `max_tool_result_chars`.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Read the existing section**

Run: `sed -n '90,120p' docs/retrieval.md`

This section already documents the timeout boundary, the read-only gating, the call bound, and the "tool failures are operational signals rather than fatal answer errors" contract. The new text must sit with those and match their voice.

- [ ] **Step 2: Add the size bound to that section**

Insert this paragraph immediately after the paragraph ending "not cancellation of
synchronous work." and before the `### Result and operational metadata` heading:

```markdown
Tool results are serialized incrementally and rejected once the encoding exceeds
`max_result_chars` (default 8192 characters). Every chunk is counted before being
appended and `text` is joined only from counted chunks, so no result over the cap
ever reaches the prompt; total tool evidence is bounded at `max_calls ×
max_result_chars`. Rejection is deliberate rather than truncation: truncated JSON
is not valid JSON, and a truncated result can drop a negation and become evidence
that misleads the verifier. An oversized result is reported with the existing
`failed` status and degrades to retrieval-only answering, exactly like an
invocation failure. Serialization is synchronous and therefore covered by no
timeout. The pure-Python encoder backing this check yields one chunk per scalar,
so the size check can only run between chunks: encoding work is bounded by the cap
plus at most one fully-encoded scalar, and peak memory and event-loop time are
proportional to the largest individual scalar in the result, not to the cap — the
same species of caveat as the synchronous selector above, whose timeout bounds
pipeline latency but does not stop the underlying thread's work.
```

- [ ] **Step 3: Verify no other docs contradict the new text**

Run: `grep -rn "max_tool_calls\|tool_timeout_seconds\|collect_tool_evidence" docs/`

Expected: any file that documents `collect_tool_evidence`'s bounds now also mentions the size cap, or does not enumerate bounds at all. Fix any file that lists the bounds but omits this one.

- [ ] **Step 4: Commit**

```bash
git add docs/retrieval.md
git commit -m "docs: document the tool result size bound"
```

---

## Completion Gate

Before claiming completion:

1. `git status --short` contains no unintended files.
2. `env HF_HOME=/tmp/agentic-search-hf-cache python3 -m pytest -q tests/unit` passes with no failures.
3. `ruff check`, `ruff format --check`, and `git diff --check` all pass over every modified file.
4. `test_read_only_tools_supply_stable_normalized_evidence` passes unmodified — the guard proving the `iterencode` swap altered no existing output.
5. `test_bounded_encoding_aborts_before_walking_the_whole_structure` passes — the guard against `_one_shot=True`.
6. Write the task report per `docs/development/self-review-reports.md` and validate it with `python3 examples/validate_task_report.py REPORT_FILE --require-tdd`.
7. Perform an independent whole-branch review against the merge base; fix every Critical or Important finding, then re-review.
