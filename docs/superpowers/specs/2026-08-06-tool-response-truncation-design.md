# Structure-aware tool-response truncation

**Date:** 2026-08-06
**Status:** Approved for planning

## Problem

`ToolAgentLoop` caps each tool message at `max_tool_response_length = 2048`
characters and truncates with `tool_response_truncate_side = "right"`, which
keeps the **end** of the string:

```python
if side == "right":
    return "(truncated)..." + text[-limit:]
```

Two defects follow, and both hit the tools that matter most.

**The model loses the best results.** Every ranked tool output — the corpus
search, `web_search`, `search_arxiv` (which requests `sortOrder=descending`),
`search_wikipedia` — is ordered best-first. Keeping the tail discards exactly
the top-ranked items and grounds the answer in the weakest ones.

**The model receives invalid JSON.** Citeable tools return a JSON array. A
character slice starts mid-object, so the model must interpret a fragment that
no parser would accept. Small local models handle this badly.

Neither defect is visible in the UI. Source cards are built from
`ToolExecutionResult.result`, which is never truncated, so the cards render
correctly while the answer is grounded in the leftovers. This is how the
condition survived unnoticed.

`_truncate_tool_response` has no direct test coverage today. The only file in
the suite referencing the cap is `tests/unit/test_public_data_knowledge.py`,
which asserts that tool output *fits* the cap rather than testing what happens
when it does not.

## Non-goals

- **No LLM summarization.** The loop's only model is `server_manager` — the
  same local model driving the agent. A summarization generation per oversized
  result would sit on the critical path of a surface that is already slow, and
  `ToolAgentLoop` is also the GRPO rollout loop, where the cost multiplies per
  rollout. The goal here is fidelity, not compression.
- No change to `max_tool_response_length`. Raising it only moves the threshold
  and spends rollout budget the answer needs.
- No per-tool custom formatters.
- No change to `ToolExecutionResult.result`, which stays untruncated so source
  cards keep full content.
- No renaming of the `"left"` / `"right"` config values (see below).

## Design

`_truncate_tool_response` gains a structure-aware path ahead of the existing
slicing:

```python
def _truncate_tool_response(self, text: str) -> str:
    limit = self.tool_config.max_tool_response_length
    if len(text) <= limit:
        return text
    fitted = _fit_json_array(text, limit)      # None when not a JSON array
    if fitted is not None:
        return fitted
    return _slice_text(text, limit, self.tool_config.tool_response_truncate_side)
```

`_fit_json_array(text, limit)` parses the content. When it is a non-empty JSON
**list**, it accumulates items from the front while the re-serialized array plus
its footer still fits, and returns valid JSON followed by:

```
...N of M results shown, K omitted for length.
```

The footer is budgeted **inside** `limit`, not appended on top of it, so the
returned string never exceeds the cap.

`_fit_json_array` returns `None` — falling back to slicing — in every other
case: a JSON object rather than a list, an empty list, prose, malformed JSON,
or a first item so large that it alone exceeds the cap. That last case
deliberately prefers a valid-prefix slice of something over a syntactically
valid empty array, which would tell the model nothing.

### Default flip

`tool_response_truncate_side` default changes from `"right"` to `"left"`, so
the fallback keeps the head. This governs prose responses such as
`web_search`'s `format_search_pages` output, which is also ranked best-first
and loses its top results today for the same reason.

The `"left"` / `"right"` values are **not** renamed, even though they read
backwards (`"right"` means keep the end). Renaming would silently change
behavior for any caller that sets them explicitly, and the existing comment at
the field already documents the semantics.

### Blast radius

This lands on `ToolAgentLoopConfig`'s default, so GRPO rollouts change shape
alongside serving. That is deliberate: a rollout that trains on amputated tool
output teaches the model from evidence whose best results were cut away, and a
serving/training mismatch in this exact path would be painful to diagnose
later. The PR states the change explicitly.

## Testing

Direct unit tests on `_truncate_tool_response`, which has none today:

1. JSON array over the cap — output parses as valid JSON, contains the **first**
   items in order, total length ≤ cap, and the footer's shown/omitted counts are
   accurate.
2. JSON array under the cap — returned byte-identical, no footer.
3. Non-JSON prose over the cap — head kept, tail dropped (the new default).
4. JSON object rather than a list — falls back to slicing without raising.
5. Malformed JSON — falls back to slicing without raising.
6. A single item larger than the whole cap — falls back to slicing; never
   returns an empty array.
7. The existing cap-fit tests in `tests/unit/test_public_data_knowledge.py`
   still pass.

## Success criteria

1. `pytest` passes, including the new tests.
2. `ruff check . && ruff format --check .` clean.
3. A `search_arxiv` result of 5 papers over the cap yields valid JSON containing
   papers 1..N in rank order, not the trailing fragment.
4. No new dependencies.
