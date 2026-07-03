# Split the HTML-to-text parser out of the search tool — design

**Date:** 2026-07-03
**Status:** Approved (design).
**Scope:** `src/tools/search.py` only. Extract the self-contained HTML→text
parsing into its own module. No change to the tool framework, `ToolAgentLoop`,
or any search behavior.

## Problem

`src/tools/search.py` is a 676-LOC kitchen sink: query sanitisation, four web
search providers (Google / SerpAPI / Serper / retrieval), tool entry points,
URL fetching, result formatting, secret redaction, **and ~80 LOC of generic
HTML-to-text parsing** (`_html_to_text`, `_SemanticTextParser`,
`_html_to_text_stdlib`).

The HTML→text block is the one cleanly-separable concern: it converts raw HTML
to readable text (BeautifulSoup with a stdlib `HTMLParser` fallback), has **no
search-specific state**, and is referenced from exactly one place inside the
module (`fetch_url`, line 345) plus two tests. It doesn't belong in the search
tool.

## Change (surgical, behavior-preserving)

1. **New module `src/tools/html_text.py`** — move `_html_to_text`,
   `_SemanticTextParser`, and `_html_to_text_stdlib` verbatim, with
   `from html.parser import HTMLParser`.
2. **`src/tools/search.py`** — delete the three moved definitions; drop the now
   unused `from html.parser import HTMLParser` import; add
   `from .html_text import _html_to_text`. The single caller (`fetch_url`)
   is unchanged.
3. **`tests/unit/test_search_tools.py`** — repoint the two
   `from src.tools.search import _html_to_text` imports to `src.tools.html_text`.

Function names are kept as-is (`_html_to_text` etc.) — this is a move, not a
rename, so behavior and call sites stay identical.

## Out of scope (deliberately)

- **Secret redaction** (`_redact_secret_params`/`_redact_url`) stays in
  `search.py`: it is called from four search-provider error paths and keys off
  the tool's own result formatting (`url=URL('…`) — coupled, not generic.
- The two same-named `OpenAPISchema` classes (`api.py` vs `openapi_schema.py`)
  are a separate clarity issue, not touched here.
- `src/internal/tools/` (`ChatTool`) is a distinct tool system — untouched.

## Testing

- Behavior-preserving: `tests/unit/test_search_tools.py` passes with the two
  import lines repointed; the HTML-parser tests exercise the moved code
  unchanged.
- `ruff check` clean (no unused imports left in `search.py`).
- `python -c "import src"` and the search-tool import surface resolve.

## Files touched

- **Add:** `src/tools/html_text.py`.
- **Modify:** `src/tools/search.py`, `tests/unit/test_search_tools.py`.
