# Generated Context Pack

# Centralize Citation Labels

## Sources

- [Specification: 2026-07-09-centralize-citation-labels-design.md](../specs/2026-07-09-centralize-citation-labels-design.md)
- [Plan: 2026-07-09-centralize-citation-labels.md](../plans/2026-07-09-centralize-citation-labels.md)

## Specification Context

### Overview

Date: 2026-07-09
Status: Approved (brainstorming)
Branch/PR: feat/centralize-citation-labels
Related: [[project_chat_orchestration]] (gap #5)

## Implementation Plan Context

### Task 1: Add helpers + rewire all sites + tests

**Files:**
- Modify: `src/context/search.py`, `src/agents/search/search.py`, `src/agents/generation/single_turn.py`
- Test: `tests/unit/test_citation_labels.py`

**Interfaces:**
- Produces (module-level in `src/context/search.py`):
  - `citation_prefix(round_idx: int, query_idx: int) -> str` → `"R{round}Q{query}D"`.
  - `citation_key(round_idx: int, query_idx: int, doc_idx: int) -> str` → `"R{round}Q{query}D{doc}"`, defined as `f"{citation_prefix(round_idx, query_idx)}{doc_idx}"`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_citation_labels.py`:

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_citation_labels.py -v`

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
