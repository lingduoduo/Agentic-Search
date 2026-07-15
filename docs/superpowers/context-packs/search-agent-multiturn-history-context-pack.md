# Generated Context Pack

# Search Agent Multiturn History

## Sources

- [Specification: 2026-07-09-search-agent-multiturn-history-design.md](../archive/specs/2026-07-09-search-agent-multiturn-history-design.md)
- [Plan: 2026-07-09-search-agent-multiturn-history.md](../archive/plans/2026-07-09-search-agent-multiturn-history.md)

## Specification Context

### Overview

Date: 2026-07-09
Status: Approved (brainstorming)
Branch/PR: feat/search-agent-multiturn-history
Related: [[project_chat_orchestration]]

## Implementation Plan Context

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
