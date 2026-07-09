# Search-Agent Multi-Turn History Implementation Plan

> Use superpowers:test-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Thread prior session Q&A into `SearchAgentLoop` (capped to 6 messages) so search-agent web mode is conversation-aware.

**Architecture:** All in `src/internal/servers/web/app.py`: a constant, a pure `_build_search_agent_messages` helper, a `history` param on `_run_search_agent`, and history wired at its two call sites.

**Tech Stack:** Python, FastAPI web backend.

## Global Constraints

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
    # Last SEARCH_AGENT_HISTORY_MESSAGES prior turns, then the query.
    assert len(msgs) == SEARCH_AGENT_HISTORY_MESSAGES + 1
    assert msgs[-1] == {"role": "user", "content": "q"}
    assert [m["content"] for m in msgs[:-1]] == [
        f"m{i}" for i in range(20 - SEARCH_AGENT_HISTORY_MESSAGES, 20)
    ]


def test_cap_default_is_six():
    assert SEARCH_AGENT_HISTORY_MESSAGES == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_search_agent_history.py -v`
Expected: FAIL — `ImportError` for `_build_search_agent_messages` / `SEARCH_AGENT_HISTORY_MESSAGES`.

- [ ] **Step 3: Write minimal implementation**

In `src/internal/servers/web/app.py`, beside `MAX_HISTORY_MESSAGES = 40` and `_trim_history`, add:

```python
# Search mode stacks long <information> observations on top of history each
# turn, so cap threaded history tighter than MAX_HISTORY_MESSAGES.
SEARCH_AGENT_HISTORY_MESSAGES = 6


def _build_search_agent_messages(query: str, history: list) -> list[dict[str, str]]:
    """Build the SearchAgentLoop message buffer: capped prior turns + the query.

    History is capped to the last ``SEARCH_AGENT_HISTORY_MESSAGES`` messages and
    mapped to ``{"role", "content"}`` dicts; the current user query is appended
    last. ``SearchAgentLoop._with_system_prompt`` prepends the system prompt.
    """
    capped = _trim_history(history, max_messages=SEARCH_AGENT_HISTORY_MESSAGES)
    messages = [{"role": m.role, "content": m.content} for m in capped]
    messages.append({"role": "user", "content": query})
    return messages
```

(Place `SEARCH_AGENT_HISTORY_MESSAGES` after `_trim_history` is defined, or define
`_build_search_agent_messages` below `_trim_history` so the reference resolves.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_search_agent_history.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/internal/servers/web/app.py tests/unit/test_search_agent_history.py
git commit -m "feat(web): _build_search_agent_messages helper + history cap

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

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

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

- Constant + pure helper (spec §Components 1–2) → Task 1. ✓
- `_run_search_agent` history param + helper call (spec §Components 3) → Task 2 Steps 1. ✓
- Both call sites threaded (spec §Components 4) → Task 2 Step 2. ✓
- Success criteria (cap, query-last, empty history, role mapping) → Task 1 tests. ✓
- No model/DB in tests (Global Constraints) → Task 1 uses `ChatMessage` fixtures only. ✓
- Types consistent: `_build_search_agent_messages(query, history) -> list[dict]` identical across Task 1 def, its tests, and the Task 2 call site. ✓
