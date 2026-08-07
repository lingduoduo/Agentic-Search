# Structure-Aware Tool-Response Truncation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `ToolAgentLoop` from handing the model an unparseable fragment
containing a ranked tool result's *worst* entries.

**Architecture:** All truncation logic moves into module-level pure functions in
`src/agents/tool/tool_calling.py`, so it is testable without constructing a loop
(which needs a tokenizer and a server manager). `_truncate_tool_response` becomes
a one-line method that reads config and delegates. A JSON array is trimmed by
dropping whole trailing items; everything else falls back to the existing
character slicing, whose default flips from keeping the tail to keeping the head.

**Tech Stack:** Python 3, stdlib `json` (already imported in the target file),
pytest.

**Spec:** `docs/superpowers/specs/2026-08-06-tool-response-truncation-design.md`

## Global Constraints

- **No new dependencies.** `json` is already imported at
  `src/agents/tool/tool_calling.py:37`.
- **No LLM call anywhere in this change.** The goal is fidelity, not compression.
- **No change to `max_tool_response_length`** (stays `2048`).
- **No change to `ToolExecutionResult.result`**, which must stay untruncated so
  source cards keep full content.
- **Do not rename the `"left"` / `"right"` / `"middle"` config values.** They read
  backwards (`"right"` means keep the end), but renaming would silently change
  behavior for any caller that sets them explicitly.
- The footer is budgeted **inside** `limit` on the JSON path — the returned
  string never exceeds `limit`.
- Style: module-level helpers are private (`_`-prefixed); follow the file's
  existing conventions.
- Lint gate for every commit: `ruff check . --fix && ruff format .`

---

### Task 1: `_fit_json_array` — trim a JSON array to the items that fit

**Files:**
- Modify: `src/agents/tool/tool_calling.py` (add two module-level functions near
  the other module-level helpers, above `class ToolAgentLoopConfig`)
- Test: `tests/unit/test_tool_response_truncation.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces, used by Task 2:
  - `_truncation_footer(shown: int, total: int, omitted: int) -> str`
  - `_fit_json_array(text: str, limit: int) -> str | None` — returns trimmed
    JSON, or `None` when *text* is not a non-empty JSON list or not even one
    item fits.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_tool_response_truncation.py`:

```python
"""Unit tests for tool-response truncation helpers."""

from __future__ import annotations

import json

from src.agents.tool.tool_calling import _fit_json_array, _truncation_footer


def test_footer_reports_counts():
    assert (
        _truncation_footer(2, 5, 3)
        == "\n...2 of 5 results shown, 3 omitted for length."
    )


def test_returns_none_for_non_json():
    assert _fit_json_array("1. Some Page\nA prose summary.", 10) is None


def test_returns_none_for_malformed_json():
    assert _fit_json_array('[{"a": 1}, {"b":', 10) is None


def test_returns_none_for_json_object():
    assert _fit_json_array('{"temperature": 14.2}', 5) is None


def test_returns_none_for_empty_list():
    assert _fit_json_array("[]", 1) is None


def test_returns_none_when_not_even_one_item_fits():
    """An empty array would be valid JSON but tells the model nothing."""
    text = json.dumps([{"a": "X" * 500}])
    assert _fit_json_array(text, 50) is None


def test_keeps_the_first_items_that_fit():
    items = [{"a": "X" * 100} for _ in range(5)]
    limit = 400

    result = _fit_json_array(json.dumps(items), limit)

    assert result is not None
    assert len(result) <= limit
    body, separator, footer = result.partition("\n...")
    assert separator, "expected a footer when items were dropped"
    kept = json.loads(body)  # must be valid JSON
    assert 1 <= len(kept) < len(items)
    # A prefix, in original order — the top-ranked items, not the tail.
    assert kept == items[: len(kept)]
    assert footer == (
        f"{len(kept)} of {len(items)} results shown, "
        f"{len(items) - len(kept)} omitted for length."
    )


def test_compacting_alone_can_make_everything_fit():
    """Indented input can shrink under the limit once re-serialized."""
    items = [{"a": 1}, {"b": 2}]
    text = json.dumps(items, indent=4)
    limit = len(text) - 1
    assert len(json.dumps(items)) <= limit

    result = _fit_json_array(text, limit)

    assert result == json.dumps(items)
    assert "omitted for length" not in result
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_tool_response_truncation.py -v`
Expected: FAIL — `ImportError: cannot import name '_fit_json_array'`

- [ ] **Step 3: Implement the two helpers**

In `src/agents/tool/tool_calling.py`, add these immediately above
`@dataclass(frozen=True)` / `class ToolAgentLoopConfig`:

```python
def _truncation_footer(shown: int, total: int, omitted: int) -> str:
    """Tell the model what it is not seeing, in its own message."""
    return f"\n...{shown} of {total} results shown, {omitted} omitted for length."


def _fit_json_array(text: str, limit: int) -> str | None:
    """Trim a JSON array to the leading items that fit within *limit*.

    Ranked tool results are ordered best-first, so dropping items from the end
    keeps what matters and leaves the model valid JSON rather than a fragment
    that starts mid-object.

    Returns None when *text* is not a non-empty JSON list, or when not even one
    item fits. The caller then falls back to character slicing: a readable
    prefix of one large item beats a valid but empty array.
    """
    try:
        items = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(items, list) or not items:
        return None

    compact = json.dumps(items)
    if len(compact) <= limit:
        # Re-serializing without the original's whitespace was enough. This
        # must be checked before the loop below: that loop charges every
        # candidate the cost of a footer, so a small indented array could
        # otherwise be rejected outright even though all of it fits.
        return compact

    total = len(items)
    kept: list[Any] = []
    for item in items:
        candidate = kept + [item]
        footer = _truncation_footer(len(candidate), total, total - len(candidate))
        if len(json.dumps(candidate) + footer) > limit:
            break
        kept = candidate

    if not kept:
        return None
    return json.dumps(kept) + _truncation_footer(len(kept), total, total - len(kept))
```

`Any` is already imported in this file; confirm with
`grep -n "^from typing" src/agents/tool/tool_calling.py` and add it to that
import if it is missing.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_tool_response_truncation.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/agents/tool/tool_calling.py tests/unit/test_tool_response_truncation.py
git commit -m "feat(tools): add _fit_json_array to trim JSON arrays by whole items"
```

---

### Task 2: Wire it in and flip the fallback default

**Files:**
- Modify: `src/agents/tool/tool_calling.py:110` (the config default) and
  `:193-203` (`_truncate_tool_response`)
- Test: `tests/unit/test_tool_response_truncation.py` (append)

**Interfaces:**
- Consumes from Task 1: `_fit_json_array(text, limit) -> str | None`.
- Produces:
  - `_slice_text(text: str, limit: int, side: str) -> str`
  - `_truncate_tool_text(text: str, limit: int, side: str) -> str`
  - `ToolAgentLoop._truncate_tool_response(self, text: str) -> str` delegates to
    `_truncate_tool_text` using `max_tool_response_length` and
    `tool_response_truncate_side`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_tool_response_truncation.py`:

```python
def test_text_under_the_limit_is_returned_unchanged():
    from src.agents.tool.tool_calling import _truncate_tool_text

    text = '[{"a": 1}]'
    assert _truncate_tool_text(text, 100, "left") == text


def test_json_array_over_the_limit_keeps_the_leading_items():
    from src.agents.tool.tool_calling import _truncate_tool_text

    items = [{"rank": i, "a": "X" * 100} for i in range(5)]

    result = _truncate_tool_text(json.dumps(items), 400, "left")

    body, _, _footer = result.partition("\n...")
    kept = json.loads(body)
    assert [item["rank"] for item in kept] == list(range(len(kept)))
    assert len(kept) < len(items)


def test_prose_over_the_limit_keeps_the_head_by_default():
    from src.agents.tool.tool_calling import _truncate_tool_text

    text = "FIRST" + "." * 100 + "LAST"

    result = _truncate_tool_text(text, 50, "left")

    assert result.startswith("FIRST")
    assert "LAST" not in result
    assert result.endswith("...(truncated)")


def test_right_side_still_keeps_the_tail():
    """The knob keeps working for anyone who sets it explicitly."""
    from src.agents.tool.tool_calling import _truncate_tool_text

    text = "FIRST" + "." * 100 + "LAST"

    result = _truncate_tool_text(text, 50, "right")

    assert result.startswith("(truncated)...")
    assert result.endswith("LAST")


def test_middle_side_keeps_both_ends():
    from src.agents.tool.tool_calling import _truncate_tool_text

    text = "FIRST" + "." * 100 + "LAST"

    result = _truncate_tool_text(text, 50, "middle")

    assert result.startswith("FIRST")
    assert result.endswith("LAST")
    assert "...(truncated)..." in result


def test_json_object_over_the_limit_falls_back_to_slicing():
    from src.agents.tool.tool_calling import _truncate_tool_text

    text = json.dumps({"temperature": 14.2, "note": "Y" * 200})

    result = _truncate_tool_text(text, 50, "left")

    assert result.startswith('{"temperature"')
    assert result.endswith("...(truncated)")


def test_the_default_truncation_side_keeps_the_head():
    """A ranked tool result must not lose its top entries."""
    from src.agents.tool.tool_calling import ToolAgentLoopConfig

    assert ToolAgentLoopConfig().tool_response_truncate_side == "left"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_tool_response_truncation.py -v`
Expected: FAIL — `ImportError: cannot import name '_truncate_tool_text'`, and
`test_the_default_truncation_side_keeps_the_head` fails with
`assert 'right' == 'left'`

- [ ] **Step 3: Add the two module-level functions**

In `src/agents/tool/tool_calling.py`, directly below `_fit_json_array` from
Task 1:

```python
def _slice_text(text: str, limit: int, side: str) -> str:
    """Character-slice *text*, keeping the side named by the config."""
    if side == "left":
        return text[:limit] + "...(truncated)"
    if side == "right":
        return "(truncated)..." + text[-limit:]
    half = limit // 2
    return text[:half] + "...(truncated)..." + text[-half:]


def _truncate_tool_text(text: str, limit: int, side: str) -> str:
    """Bound one tool response, preferring whole JSON items over a raw slice."""
    if len(text) <= limit:
        return text
    fitted = _fit_json_array(text, limit)
    if fitted is not None:
        return fitted
    return _slice_text(text, limit, side)
```

- [ ] **Step 4: Replace the method body**

Replace the existing method at `src/agents/tool/tool_calling.py:193-203`:

```python
    def _truncate_tool_response(self, text: str) -> str:
        limit = self.tool_config.max_tool_response_length
        if len(text) <= limit:
            return text
        side = self.tool_config.tool_response_truncate_side
        if side == "left":
            return text[:limit] + "...(truncated)"
        if side == "right":
            return "(truncated)..." + text[-limit:]
        half = limit // 2
        return text[:half] + "...(truncated)..." + text[-half:]
```

with:

```python
    def _truncate_tool_response(self, text: str) -> str:
        return _truncate_tool_text(
            text,
            self.tool_config.max_tool_response_length,
            self.tool_config.tool_response_truncate_side,
        )
```

- [ ] **Step 5: Flip the default and update its comment**

At `src/agents/tool/tool_calling.py:106-110`, replace:

```python
    # How to truncate a tool response that exceeds max_tool_response_length:
    #   "left"   — keep the start, append "...(truncated)"
    #   "right"  — prepend "(truncated)...", keep the end
    #   "middle" — keep equal halves from start and end
    tool_response_truncate_side: str = "right"
```

with:

```python
    # Fallback policy for a tool response that exceeds
    # max_tool_response_length and is NOT a JSON array (arrays are trimmed by
    # whole items instead — see _fit_json_array). Defaults to keeping the
    # start: tool results are ranked best-first, so dropping the tail loses
    # the least.
    #   "left"   — keep the start, append "...(truncated)"
    #   "right"  — prepend "(truncated)...", keep the end
    #   "middle" — keep equal halves from start and end
    tool_response_truncate_side: str = "left"
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/unit/test_tool_response_truncation.py -v`
Expected: PASS (15 tests total across both tasks)

- [ ] **Step 7: Run the tests that depend on this behavior**

Run: `pytest tests/unit/test_public_data_knowledge.py tests/unit/test_tool_backend.py tests/unit/test_tool_agent_runner.py tests/unit/test_tool_error_feedback.py -v`
Expected: PASS. These include the cap-fit tests added in PR #503, which assert
tool output fits within `max_tool_response_length`.

- [ ] **Step 8: Run the full suite**

Run: `python3 -m pytest -q 2>&1 | tail -5`
Expected: PASS with no failures. Run it in the FOREGROUND, not as a background
job. If a test fails because it asserted the old tail-keeping behavior, that
assertion is now describing the defect this plan fixes — update it and say so
explicitly in your report. If a test fails for any other reason, investigate
and report BLOCKED rather than weakening the assertion.

- [ ] **Step 9: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/agents/tool/tool_calling.py tests/unit/test_tool_response_truncation.py
git commit -m "fix(tools): trim tool responses by whole items, keeping the best results"
```

---

## Manual verification

Not required for merge — the unit tests cover the behavior — but if the local
stack is already running, asking `/tools` for "ArXiv papers about dense
retrieval" should now show an answer that discusses the *first* papers in the
result rather than the last, and the tool trace's message should parse as JSON.
