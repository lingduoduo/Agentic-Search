# Centralize Citation-Label Contract Implementation Plan

> Use superpowers:test-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `citation_prefix` / `citation_key` in `src/context/search.py` the single source of the `R{r}Q{q}D{d}` format, rewire all six derivation sites, and lock formatter↔parser agreement with a round-trip test.

**Architecture:** Two helpers in `context/search.py` (`citation_key` defined via `citation_prefix`); rewire `context/search.py` (`_citation_keys`, three `cited_*` methods), `agents/search/search.py` (`_format_round_information`), and `agents/generation/single_turn.py` (two sites). Behavior-preserving.

**Tech Stack:** Python.

## Global Constraints

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
    agent_ctx.rounds.append([ctx])
    agent_ctx.turns.append(ctx)
    assert agent_ctx.cited_result_ids("cite [R1Q2D1]") == frozenset({"R1Q2D1"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_citation_labels.py -v`
Expected: FAIL — `ImportError` for `citation_key` / `citation_prefix`.

- [ ] **Step 3: Add the helpers**

In `src/context/search.py`, just below `_CITATION_RE` / `_citation_keys` (top of module), add:

```python
def citation_prefix(round_idx: int, query_idx: int) -> str:
    """Citation-label prefix for a (round, query): ``R{round}Q{query}D``."""
    return f"R{round_idx}Q{query_idx}D"


def citation_key(round_idx: int, query_idx: int, doc_idx: int) -> str:
    """Full citation key ``R{round}Q{query}D{doc}`` (the string inside ``[...]``)."""
    return f"{citation_prefix(round_idx, query_idx)}{doc_idx}"
```

Then rewire `_citation_keys` reconstruction:
```python
def _citation_keys(answer_text: str) -> set[str]:
    return {
        citation_key(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        for m in _CITATION_RE.finditer(answer_text)
    }
```

- [ ] **Step 4: Rewire the three `cited_*` methods**

In `cited_result_ids` (:182), `cited_results` (:196), `cited_search_contexts` (:209),
replace each inline `f"R{round_idx}Q{query_idx}D{doc_idx}"` with
`citation_key(round_idx, query_idx, doc_idx)`.

- [ ] **Step 5: Rewire the formatter call sites**

- `src/agents/search/search.py:_format_round_information` (:668): import
  `citation_prefix` (add to the existing `from src.context.search import ...`)
  and replace `citation_prefix=f"R{round_index}Q{i}D"` with
  `citation_prefix=citation_prefix(round_index, i)`.
- `src/agents/generation/single_turn.py` (:209, :224): import `citation_prefix`
  and replace `citation_prefix="R1Q1D"` with `citation_prefix=citation_prefix(1, 1)`.

(Watch for a local-name clash: the keyword arg `citation_prefix=` and the function
`citation_prefix` share a name but do not collide — the keyword is `to_information_block`'s
parameter; the value is the function call. Confirmed fine.)

- [ ] **Step 6: Run new tests + full behavior regression**

Run: `python3 -m pytest tests/unit/test_citation_labels.py tests/unit/test_agent_loop.py tests/unit/test_reward.py tests/unit/test_reward_shapes.py tests/unit/test_components.py -q`
Expected: PASS — new tests green and every existing citation/reward test unchanged
(byte-identical labels).

- [ ] **Step 7: Grep-verify no stray inline format remains**

Run: `grep -rn 'f"R{.*}Q{.*}D' src/context/search.py src/agents/search/search.py src/agents/generation/single_turn.py`
Expected: only the definition inside `citation_prefix` (no other inline `f"R{…}Q{…}D…"`).

- [ ] **Step 8: Commit**

```bash
git add src/context/search.py src/agents/search/search.py src/agents/generation/single_turn.py tests/unit/test_citation_labels.py
git commit -m "refactor(context): centralize citation-label format into helpers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

- Helpers `citation_prefix`/`citation_key` (spec §Approach) → Step 3 + `test_prefix_and_key_format`, `test_key_is_prefix_plus_doc`. ✓
- `_citation_keys` + three `cited_*` rewired → Steps 3–4. ✓
- Formatter sites (search.py, single_turn.py ×2) → Step 5. ✓
- `_CITATION_RE` stays sole parse regex; reward.py untouched → Global Constraints + Step 6 (reward tests). ✓
- Round-trip guard (spec success criteria) → `test_round_trip_parse_recovers_key`, `test_formatter_and_parser_agree_end_to_end`. ✓
- No stray inline format → Step 7. ✓
- Types consistent: `citation_prefix(int,int)->str`, `citation_key(int,int,int)->str` identical across def, tests, and call sites. ✓
