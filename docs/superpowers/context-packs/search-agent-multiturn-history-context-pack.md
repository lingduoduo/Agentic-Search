# Generated Context Pack

# Search Agent Multiturn History

## Sources

- [Specification: 2026-07-09-search-agent-multiturn-history-design.md](../specs/2026-07-09-search-agent-multiturn-history-design.md)
- [Plan: 2026-07-09-search-agent-multiturn-history.md](../plans/2026-07-09-search-agent-multiturn-history.md)

## Specification Context

### Non-goals

- No change to the base token-crop (`base.py:186` keep-tail is non-message-aware;
  that is a separate, known gap — out of scope).
- No change to `_run_tool_agent`'s existing history handling or the RAG path.
- No summarization / semantic compression of history.
- No new persistence — history already loads from `AgenticSearchStore` and is
  trimmed to `MAX_HISTORY_MESSAGES = 40` in `_run_agent_impl` (`app.py:1267`).

### Components (all in `src/internal/servers/web/app.py`)

1. **Constant** — `SEARCH_AGENT_HISTORY_MESSAGES = 6` (last 3 user/assistant
   exchanges), beside `MAX_HISTORY_MESSAGES`. Tighter than 40 because search mode
   adds its own long observations each turn, and the base token-crop drops from
   the front (the system prompt) on overflow.

2. **Pure helper** — `_build_search_agent_messages(query: str, history: list) -> list[dict]`:
   - `capped = _trim_history(history, max_messages=SEARCH_AGENT_HISTORY_MESSAGES)`
   - map each entry to `{"role": m.role, "content": m.content}`
   - append `{"role": "user", "content": query}`
   - return the list. No model/loop dependency → unit-testable in isolation.

3. **`_run_search_agent`** — add a `history: list` keyword parameter; replace the
   inline `[{"role": "user", "content": query}]` with
   `_build_search_agent_messages(query, history)` passed to `loop.run(...)`.

4. **Call sites** — `app.py:839` and `app.py:1459` pass `history=history`
   (already in scope at both).

### Testing

Unit tests (no model load — respects the web-test model-load gotcha) on the pure
helper `_build_search_agent_messages`:
1. Empty history → `[{"role": "user", "content": query}]`.
2. Short history (< cap) → all prior turns + query, in order, query last.
3. Long history (> cap) → only the last `SEARCH_AGENT_HISTORY_MESSAGES` prior
   turns + query.
4. Role/content mapping preserved from `ChatMessage`-like entries.

The `_run_search_agent` wiring (loop construction) is covered by the existing web
integration tests; the new behavior worth asserting in isolation is the message
buffer construction, which the pure helper localizes.

### Risks

- **Token budget**: history + growing observations could still overflow and the
  base crop would drop the system prompt. Mitigated (not eliminated) by the tight
  cap of 6; the deeper fix (message-aware crop) is deliberately out of scope.

## Implementation Plan Context

### Global Constraints

- Branch off `main` (never commit to `main`); branch `feat/search-agent-multiturn-history`.
- Only touch `_run_search_agent` and the message-buffer construction; do NOT change `_run_tool_agent`, the RAG path, or the base token-crop.
- History items are `src.context.ChatMessage` with `.role` / `.content`.
- Tests must NOT load a model or DB (web-test model-load gotcha) — test the pure helper.
- Match repo ruff formatting.

---

### Task 1: Pure helper `_build_search_agent_messages` + constant

**Files:**
- Modify: `src/internal/servers/web/app.py`
- Test: `tests/unit/test_search_agent_history.py`

**Interfaces:**
- Produces: `SEARCH_AGENT_HISTORY_MESSAGES: int = 6`; `_build_search_agent_messages(query: str, history: list) -> list[dict[str, str]]` returning `[…≤6 capped prior turns as {"role","content"}…, {"role": "user", "content": query}]`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_search_agent_history.py`:

```python
"""Unit tests for search-agent conversation-history message building."""

from __future__ import annotations

from src.context import ChatMessage
from src.internal.servers.web.app import (
    SEARCH_AGENT_HISTORY_MESSAGES,
    _build_search_agent_messages,
)


def test_empty_history_yields_just_the_query():
    msgs = _build_search_agent_messages("what is FAISS?", [])
    assert msgs == [{"role": "user", "content": "what is FAISS?"}]


def test_short_history_prepended_then_query_last():
    history = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
    ]
    msgs = _build_search_agent_messages("next question", history)
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "next question"},
    ]


def test_long_history_capped_to_last_n_plus_query():
    history = [
        ChatMessage(role=("user" if i % 2 == 0 else "assistant"), content=f"m{i}")
        for i in range(20)
    ]
    msgs = _build_search_agent_messages("q", history)

_[Section compacted.]_

### Task 2: Wire history into `_run_search_agent` + both call sites

**Files:**
- Modify: `src/internal/servers/web/app.py`

**Interfaces:**
- Consumes: `_build_search_agent_messages` (Task 1).
- Changes: `_run_search_agent(query, *, history: list, manager, tokenizer, search_url, top_k, on_turn=None, on_trace=None)`.

- [ ] **Step 1: Add `history` param and use the helper**

In `_run_search_agent` (`app.py:564`), add `history: list` to the keyword-only
params and replace:

```python
    output = await loop.run(
        [{"role": "user", "content": query}],
```
with:
```python
    output = await loop.run(
        _build_search_agent_messages(query, history),
```

- [ ] **Step 2: Thread history at both call sites**

At `app.py:839` and `app.py:1459`, add `history=history,` to the
`_run_search_agent(...)` call (the surrounding functions already have `history`
in scope).

- [ ] **Step 3: Verify no regressions in the web module import + helper tests**

Run: `python3 -m pytest tests/unit/test_search_agent_history.py -q`
Expected: PASS.

Run: `python3 -c "import ast, sys; ast.parse(open('src/internal/servers/web/app.py').read()); print('app.py parses')"`
Expected: `app.py parses` (syntactic sanity without importing heavy deps).

- [ ] **Step 4: Grep-verify both call sites now pass history**

Run: `grep -n "_run_search_agent(" src/internal/servers/web/app.py` and confirm
each call is followed (within its argument list) by `history=history`.

- [ ] **Step 5: Commit**

```bash
git add src/internal/servers/web/app.py
git commit -m "feat(web): thread session history into search-agent mode

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
