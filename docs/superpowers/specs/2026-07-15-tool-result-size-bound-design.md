# Bounded Tool Result Size Design

## Goal

Bound the size of a registered tool's result before it becomes RAG evidence, so that an
oversized result cannot enter the prompt, exhaust memory, or block the event loop.
Oversized results degrade to retrieval-only answering through the fail-closed path that
`collect_tool_evidence` already implements.

## Current State

`collect_tool_evidence` validates the registry, the descriptors, the selected request, the
call count, and the invocation timeout. It does not validate the tool's *result*. At
`src/context/tool_evidence.py:134-136` the result is serialized and accepted verbatim:

```python
result = await asyncio.wait_for(registry.invoke(request), timeout=timeout_seconds)
text = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
```

`text` becomes `EvidenceSource.text` (`tool_evidence.py:143-151`) and flows into the guarded
generation prompt. There is no size limit, so three distinct failures are unbounded:

1. **Prompt size.** A large result is injected into the prompt whole, consuming context and
   cost, and presenting a large injection surface.
2. **Peak memory.** `json.dumps` materializes the entire serialized string.
3. **Event-loop blocking.** `json.dumps` runs synchronously inside the coroutine *after*
   `asyncio.wait_for` has already returned. It is covered by no timeout. Serializing a very
   large result blocks the event loop for the full duration of the serialization, stalling
   every other request in the process.

The third is the reason a post-serialization length check is insufficient: by the time the
length is known, the cost has already been paid.

The module's docstring (`tool_evidence.py:69-71`) already promises that "results that cannot
be represented as JSON are ignored so retrieval-based answering can continue". This change
makes the code honor that promise for results that are too large to represent safely.

## Chosen Approach

Bound the result during encoding, using `json.JSONEncoder.iterencode`, and abort as soon as
the running length exceeds the cap.

```python
encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False)
chunks: list[str] = []
total = 0
for chunk in encoder.iterencode(result):
    total += len(chunk)
    if total > max_result_chars:
        raise ValueError("tool result exceeds max_result_chars")
    chunks.append(chunk)
text = "".join(chunks)
```

This bounds prompt size, peak memory, and event-loop blocking with one mechanism. A hostile
or defective tool causes at most `max_result_chars` of encoding work.

`iterencode` with identical `sort_keys`, `separators`, and `ensure_ascii` settings produces
output byte-identical to the current `json.dumps` call for every result under the cap, so no
existing behavior changes. Verified empirically before adoption: a mixed structure containing
nested objects, a float, and a non-ASCII string encodes to 2013 chunks whose join is exactly
equal to the current `json.dumps` output.

### Implementation constraint: do not pass `_one_shot=True`

`JSONEncoder.iterencode(o)` defaults to `_one_shot=False`, which selects the pure-Python
`_make_iterencode` generator and yields many small chunks. `json.dumps` internally calls
`iterencode(o, _one_shot=True)`, which selects the C encoder and returns the entire document
as a **single chunk**.

Early abort therefore depends on the default. If someone later "optimizes" this call by
passing `_one_shot=True`, the loop receives one chunk, the size check fires only after the
whole result is already encoded, and the memory and event-loop protections silently vanish
while every test that only checks the returned evidence still passes. Test 5 exists to fail
loudly if that happens.

The cost of the pure-Python encoder is bounded by the cap: at most `max_result_chars` of
encoding work, which at 8192 characters is negligible.

### Rejected alternative: check length after serializing

`text = json.dumps(...)` followed by `if len(text) > cap: reject` is shorter, but it bounds
only what reaches the prompt. It still materializes the whole string and still blocks the
event loop for the full serialization. It cannot address failure 3 at any cap value.

### Rejected alternative: truncate instead of reject

Truncated JSON is invalid JSON, and truncation can silently drop a negation, converting a
result into evidence that misleads the lexical verifier in `verify_answer_draft`. Rejection
matches this module's existing contract, where every failure path returns `[]` or `continue`
rather than degraded data.

## Contracts

### `collect_tool_evidence`

Add one keyword-only parameter:

- `max_result_chars: int = 8192` — maximum length, in characters, of the serialized tool
  result.

Validated with the existing guards at `tool_evidence.py:72-75`:

```python
if max_result_chars <= 0:
    raise ValueError("max_result_chars must be positive")
```

The default of 8192 is deliberately about four times `ToolAgentLoop`'s
`max_tool_response_length` of 2048 (`src/agents/tool/tool_calling.py:92`), because that path
truncates where this one rejects. A cap that merely shortens a result may sit close to the
typical result size; a cap that discards it should not.

The boundary is exclusive: a result whose serialized length is exactly `max_result_chars` is
accepted; `max_result_chars + 1` is rejected.

### `answer_with_retrieval` (`src/context/pipeline.py`)

Add `max_tool_result_chars: int = 8192` alongside the existing `max_tool_calls: int = 2`
(`pipeline.py:276`) and pass it through as `max_result_chars=max_tool_result_chars` at the
`collect_tool_evidence` call site (`pipeline.py:301`), mirroring how `max_tool_calls` is
threaded today.

## Behavior

An oversized result raises `ValueError` from inside the existing `try` block, where the
`except Exception` at `tool_evidence.py:137-140` already marks the tool `"failed"` via
`status_callback` and continues to the next request. Consequences:

- No new status value, no new branch, and no new trace attribute.
- The tool contributes no `EvidenceSource`.
- Remaining selected tools are still attempted, up to `max_calls`.
- Retrieval evidence still answers the question. If no evidence survives at all, the
  existing canonical abstention at `pipeline.py:90-94` applies unchanged.
- Tool status continues to have no influence on the answer; it is observability only.

## Testing

Test-driven, RED before GREEN, in `tests/unit/test_rag_tool_evidence.py`:

1. A result serializing above the cap yields no evidence and reports status `"failed"`.
2. A result serializing to exactly the cap is accepted and yields one `EvidenceSource`.
3. For an under-cap result, `EvidenceSource.text` is byte-identical to
   `json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
   This is the no-regression guard on the `iterencode` swap.
4. `max_result_chars=0` and a negative value each raise `ValueError`.
5. An oversized result aborts encoding early rather than serializing the whole structure.
   Asserted with a sentinel: a list of oversized strings terminated by an object json cannot
   serialize. Early abort raises the size `ValueError` without ever reaching the sentinel;
   encoding the whole structure would instead raise `TypeError`. The distinction proves the
   walk stopped early, and the test fails if `_one_shot=True` is ever introduced. Verified
   to behave as designed before adoption: the walk halts at ~9k characters and never reaches
   the sentinel.
6. An oversized result from one tool does not prevent a second, under-cap tool from
   contributing evidence.
7. Pipeline-level: guarded answering still succeeds from retrieval evidence when the only
   selected tool returns an oversized result.

## Out of Scope

Each is a separate finding and, if pursued, a separate change:

- Per-tool declared output schemas (contract-drift validation).
- Content or semantic validation of tool results. A well-formed, under-cap, factually wrong
  result remains trusted; `verify_answer_draft` validates the model's *use* of a result, not
  the result itself.
- The absent timeout on `FunctionTool.execute` (`src/tools/base.py:155`), which is a
  different tool path from `collect_tool_evidence`.
- Byte-length versus character-length accounting. The cap counts characters, consistent with
  `max_tool_response_length`.
