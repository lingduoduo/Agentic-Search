# Generated Context Pack

# Centralize Citation Labels

## Sources

- [Specification: 2026-07-09-centralize-citation-labels-design.md](../specs/2026-07-09-centralize-citation-labels-design.md)
- [Plan: 2026-07-09-centralize-citation-labels.md](../plans/2026-07-09-centralize-citation-labels.md)

## Specification Context

### Non-goals

- Do NOT touch `reward.py`'s `_CITATION_RE` (:17) — it is a **different, broader**
  pattern (`\[(?:D\d+|R\d+Q\d+D\d+)\]`) for a presence-only format-compliance
  check that intentionally also accepts compact `[D1]` labels. It is not this
  specific-key contract.
- No change to `SearchContext.to_information_block`'s signature — callers keep
  passing a `citation_prefix` string, just built via the shared helper.
- Behavior-preserving: every emitted/parsed string stays byte-identical.

### Testing (no model load)

New tests on the helpers (pure):
1. `citation_prefix(1, 2) == "R1Q2D"`; `citation_key(1, 2, 3) == "R1Q2D3"`.
2. Consistency: `citation_key(r,q,d) == citation_prefix(r,q) + str(d)` for a few tuples.
3. Round-trip: `_citation_keys(f"see [{citation_key(1,2,3)}] here") == {"R1Q2D3"}`.
4. `to_information_block(citation_prefix=citation_prefix(1,2))` emits `[R1Q2D1]`
   for the first doc, and `AgentContext.cited_result_ids` matches it (formatter ↔
   parser agree end-to-end via the helpers).

Existing `cited_result_ids`/`cited_results` tests and the reward `citation_support`
tests remain the behavior guard.

### Risks

- Low. Pure string refactor; the round-trip test locks the formatter/parser
  agreement that was previously only implicit.

## Implementation Plan Context

### Global Constraints

- Branch off `main` (never commit to `main`); branch `feat/centralize-citation-labels`.
- Behavior-preserving: every emitted/parsed citation string stays byte-identical.
- Do NOT touch `reward.py`'s `_CITATION_RE` (a different, broader presence pattern).
- Do NOT change `SearchContext.to_information_block`'s signature (callers still pass a `citation_prefix` string, built via the helper).
- Match repo ruff formatting.

---

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

```python
"""Unit tests for the centralized citation-label contract."""

from __future__ import annotations

from src.context.search import (
    AgentContext,
    SearchContext,
    SearchResult,
    _citation_keys,
    citation_key,
    citation_prefix,
)


def test_prefix_and_key_format():
    assert citation_prefix(1, 2) == "R1Q2D"
    assert citation_key(1, 2, 3) == "R1Q2D3"


def test_key_is_prefix_plus_doc():
    for r, q, d in [(1, 1, 1), (2, 3, 4), (10, 2, 7)]:
        assert citation_key(r, q, d) == citation_prefix(r, q) + str(d)


def test_round_trip_parse_recovers_key():
    key = citation_key(1, 2, 3)
    assert _citation_keys(f"grounded [{key}] here") == {key}


def test_formatter_and_parser_agree_end_to_end():
    ctx = SearchContext(
        query="voice actor",
        results=[SearchResult(contents='"Voice"\nAlice David')],
    )
    block = ctx.to_information_block(citation_prefix=citation_prefix(1, 2))
    assert "[R1Q2D1]" in block

    agent_ctx = AgentContext(tasks={})

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
