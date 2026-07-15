# Generated Context Pack

# Cap Full Page Fetch Content

## Sources

- [Specification: 2026-07-09-cap-full-page-fetch-content-design.md](../specs/2026-07-09-cap-full-page-fetch-content-design.md)
- [Plan: 2026-07-09-cap-full-page-fetch-content.md](../plans/2026-07-09-cap-full-page-fetch-content.md)

## Specification Context

### Non-goals

- No cap on search snippets (those are server-controlled via `topk`).
- No change to the base token-crop (`base.py:186`) — separate known gap.
- No change to `ToolAgentLoop`.
- No total/cross-page budget (per-page cap only — see decision below).

### Decision: per-page cap (not total budget)

Cap each page's `contents` independently. This directly fixes "a large fetched
page blows the budget" and mirrors ToolAgentLoop's per-response cap. A total
budget across all pages fetched in one turn would be more robust when many pages
are fetched at once, but adds budget-division complexity; deferred (YAGNI).

### Testing (no model load)

Unit tests on the pure helper `_truncate_page_content`:
1. `len(text) <= limit` → unchanged.
2. `len(text) > limit` → `text[:limit]` + marker; result startswith the head and
   endswith the marker.
3. `limit <= 0` → unchanged (disabled).

Plus one formatter test: `_format_full_page_information` on a `SearchResult` with
oversized `contents` (built directly, no loop/model) truncates it, while a small
page is left intact. Construct the loop with a tiny `max_full_page_chars` to
exercise the cap deterministically.

### Risks

- Truncating mid-sentence could drop relevant later content; acceptable for a
  bounded observation, and `max_full_page_chars` is tunable / disableable.

## Implementation Plan Context

### Global Constraints

- Branch off `main` (never commit to `main`); branch `feat/cap-full-page-fetch-content`.
- Only touch the full-page formatter path; not search snippets, not the base token-crop, not ToolAgentLoop.
- `SearchResult` fields: `contents: str`, `score`, `title`, `url`, `metadata`.
- Tests: pure-helper tests need no model; the formatter test constructs `SearchAgentLoop` with the existing `DummyTokenizerWithEncode` / `DummyServerManager` (from `tests/unit/test_agent_loop.py`).
- Match repo ruff formatting.

---

### Task 1: config field + `_truncate_page_content` helper + apply + tests

**Files:**
- Modify: `src/agents/search/search.py`
- Test: `tests/unit/test_full_page_truncation.py`

**Interfaces:**
- Produces:
  - `SearchAgentLoopConfig.max_full_page_chars: int = 4096`.
  - module-level `_truncate_page_content(text: str, limit: int) -> str`.
  - `_format_full_page_information` applies the helper per page.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_full_page_truncation.py`:

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_full_page_truncation.py -v`
Expected: FAIL — `ImportError` for `_truncate_page_content` / no `max_full_page_chars`.

- [ ] **Step 3: Write minimal implementation**

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
