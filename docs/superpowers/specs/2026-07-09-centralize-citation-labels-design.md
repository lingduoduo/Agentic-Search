# Centralize Citation-Label Contract — Design

Date: 2026-07-09
Status: Approved (brainstorming)
Branch/PR: feat/centralize-citation-labels
Related: [[project_chat_orchestration]] (gap #5)

## Problem

The citation-label format `R{round}Q{query}D{doc}` (e.g. `R1Q2D3`) — the implicit
contract linking the observation formatter to the reward's `citation_support`
term — is re-derived in **six** places:

- `src/context/search.py`: parse regex `_CITATION_RE` (:9), reconstruction in
  `_citation_keys` (:14), and inline `f"R{r}Q{q}D{d}"` in `cited_result_ids`
  (:182), `cited_results` (:196), `cited_search_contexts` (:209).
- `src/agents/search/search.py`: prefix `f"R{round}Q{i}D"` in
  `_format_round_information` (:668).
- `src/agents/generation/single_turn.py`: hardcoded `"R1Q1D"` prefix (:209, :224).

Any change to the format in one site would silently break the others (and the
`citation_support` reward). There is no single source of truth and no test that
the formatter and parser agree.

## Non-goals

- Do NOT touch `reward.py`'s `_CITATION_RE` (:17) — it is a **different, broader**
  pattern (`\[(?:D\d+|R\d+Q\d+D\d+)\]`) for a presence-only format-compliance
  check that intentionally also accepts compact `[D1]` labels. It is not this
  specific-key contract.
- No change to `SearchContext.to_information_block`'s signature — callers keep
  passing a `citation_prefix` string, just built via the shared helper.
- Behavior-preserving: every emitted/parsed string stays byte-identical.

## Approach

Two module-level helpers in `src/context/search.py` (already the home of the
parse side), with `citation_key` defined **in terms of** `citation_prefix` so
they cannot drift:

```python
def citation_prefix(round_idx: int, query_idx: int) -> str:
    """Citation-label prefix for a (round, query): 'R{round}Q{query}D'."""
    return f"R{round_idx}Q{query_idx}D"


def citation_key(round_idx: int, query_idx: int, doc_idx: int) -> str:
    """Full citation key 'R{round}Q{query}D{doc}' (the string inside [...])."""
    return f"{citation_prefix(round_idx, query_idx)}{doc_idx}"
```

Rewire every derivation site:
- `_citation_keys` → reconstruct via `citation_key(...)`.
- `cited_result_ids` / `cited_results` / `cited_search_contexts` → `citation_key(...)`.
- `search.py:_format_round_information` → `citation_prefix(round_index, i)`.
- `single_turn.py` (both sites) → `citation_prefix(1, 1)`.
- `_CITATION_RE` stays the single parse regex.

## Success criteria

- The `R…Q…D…` literal appears in exactly two definitions (`citation_prefix`,
  and the `_CITATION_RE` regex); no other inline `f"R{...}Q{...}D{...}"`.
- Round-trip holds: `_CITATION_RE` parses `[{citation_key(r,q,d)}]` back to
  `citation_key(r,q,d)`; and `citation_key(r,q,d) == citation_prefix(r,q) + str(d)`.
- All existing citation/reward tests stay green (byte-identical output).

## Testing (no model load)

New tests on the helpers (pure):
1. `citation_prefix(1, 2) == "R1Q2D"`; `citation_key(1, 2, 3) == "R1Q2D3"`.
2. Consistency: `citation_key(r,q,d) == citation_prefix(r,q) + str(d)` for a few tuples.
3. Round-trip: `_citation_keys(f"see [{citation_key(1,2,3)}] here") == {"R1Q2D3"}`.
4. `to_information_block(citation_prefix=citation_prefix(1,2))` emits `[R1Q2D1]`
   for the first doc, and `AgentContext.cited_result_ids` matches it (formatter ↔
   parser agree end-to-end via the helpers).

Existing `cited_result_ids`/`cited_results` tests and the reward `citation_support`
tests remain the behavior guard.

## Risks

- Low. Pure string refactor; the round-trip test locks the formatter/parser
  agreement that was previously only implicit.
