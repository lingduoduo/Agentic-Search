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

…

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

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_search_agent_history.py -v`
Expected: FAIL — `ImportError` for `_build_search_agent_messages` / `SEARCH_AGENT_HISTORY_MESSAGES`.

…

### Task 2: Wire history into `_run_search_agent` + both call sites

**Files:**
- Modify: `src/internal/servers/web/app.py`

**Interfaces:**
- Consumes: `_build_search_agent_messages` (Task 1).
- Changes: `_run_search_agent(query, *, history: list, manager, tokenizer, search_url, top_k, on_turn=None, on_trace=None)`.

- [ ] **Step 1: Add `history` param and use the helper**

In `_run_search_agent` (`app.py:564`), add `history: list` to the keyword-only
params and replace:

with:
- [ ] **Step 2: Thread history at both call sites**

At `app.py:839` and `app.py:1459`, add `history=history,` to the
`_run_search_agent(...)` call (the surrounding functions already have `history`
in scope).

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
