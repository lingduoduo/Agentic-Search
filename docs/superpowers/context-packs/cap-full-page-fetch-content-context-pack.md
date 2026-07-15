# Generated Context Pack

# Cap Full Page Fetch Content

## Sources

- [Specification: 2026-07-09-cap-full-page-fetch-content-design.md](../archive/specs/2026-07-09-cap-full-page-fetch-content-design.md)
- [Plan: 2026-07-09-cap-full-page-fetch-content.md](../archive/plans/2026-07-09-cap-full-page-fetch-content.md)

## Specification Context

### Decision: per-page cap (not total budget)

Cap each page's `contents` independently. This directly fixes "a large fetched
page blows the budget" and mirrors ToolAgentLoop's per-response cap. A total
budget across all pages fetched in one turn would be more robust when many pages
are fetched at once, but adds budget-division complexity; deferred (YAGNI).

## Implementation Plan Context

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
