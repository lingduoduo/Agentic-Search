# Generated Context Pack

# Split Search Tool Html

## Sources

- [Specification: 2026-07-03-split-search-tool-html-design.md](../specs/2026-07-03-split-search-tool-html-design.md)
- [Plan: 2026-07-03-split-search-tool-html.md](../plans/2026-07-03-split-search-tool-html.md)

## Specification Context

### Out of scope (deliberately)

- **Secret redaction** (`_redact_secret_params`/`_redact_url`) stays in
  `search.py`: it is called from four search-provider error paths and keys off
  the tool's own result formatting (`url=URL('…`) — coupled, not generic.
- The two same-named `OpenAPISchema` classes (`api.py` vs `openapi_schema.py`)
  are a separate clarity issue, not touched here.
- `src/internal/tools/` (`ChatTool`) is a distinct tool system — untouched.

## Implementation Plan Context

### Task 1: Create html_text.py and rewire search.py

- [x] **Step 1:** Create `src/tools/html_text.py` with `_html_to_text`, `_SemanticTextParser`, `_html_to_text_stdlib` (+ `from __future__ import annotations`, `from html.parser import HTMLParser`).
- [x] **Step 2:** In `search.py`, delete the three definitions; remove `from html.parser import HTMLParser`; add `from .html_text import _html_to_text`.
- [x] **Verify:** `ruff check src/tools/search.py src/tools/html_text.py` clean; `python -c "import src.tools.search"` OK.

### Task 2: Repoint tests + verify

- [x] **Step 1:** In `tests/unit/test_search_tools.py`, change the two `from src.tools.search import _html_to_text` to `from src.tools.html_text import _html_to_text`.
- [x] **Step 2:** `pytest tests/unit/test_search_tools.py` green.
- [x] **Verify:** grep confirms `_html_to_text`/`_SemanticTextParser` no longer defined in `search.py`; search.py LOC reduced (~676 → ~595).

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
