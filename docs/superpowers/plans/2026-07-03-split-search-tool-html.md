# Split Search-Tool HTML Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Move the self-contained HTML-to-text parser out of the 676-LOC `search.py` into `src/tools/html_text.py`.

**Architecture:** New `html_text.py` owns `_html_to_text` / `_SemanticTextParser` / `_html_to_text_stdlib`; `search.py` imports `_html_to_text` from it. Pure move — behavior-preserving.

**Tech Stack:** Python 3, pytest. No new deps.

**Spec:** `docs/superpowers/specs/2026-07-03-split-search-tool-html-design.md`.

## Global Constraints

- **Behavior-preserving.** Verbatim move; function names + the one caller (`fetch_url`) unchanged.
- **Scope:** the HTML block only. Redaction stays (coupled to search providers). No framework/OpenAPI/ChatTool changes.

---

## File Structure

- **Add** `src/tools/html_text.py` — the 3 moved functions/classes + `HTMLParser` import.
- **Modify** `src/tools/search.py` — delete moved defs, drop unused `HTMLParser` import, add `from .html_text import _html_to_text`.
- **Modify** `tests/unit/test_search_tools.py` — repoint 2 imports.

---

### Task 1: Create html_text.py and rewire search.py

- [x] **Step 1:** Create `src/tools/html_text.py` with `_html_to_text`, `_SemanticTextParser`, `_html_to_text_stdlib` (+ `from __future__ import annotations`, `from html.parser import HTMLParser`).
- [x] **Step 2:** In `search.py`, delete the three definitions; remove `from html.parser import HTMLParser`; add `from .html_text import _html_to_text`.
- [x] **Verify:** `ruff check src/tools/search.py src/tools/html_text.py` clean; `python -c "import src.tools.search"` OK.

### Task 2: Repoint tests + verify

- [x] **Step 1:** In `tests/unit/test_search_tools.py`, change the two `from src.tools.search import _html_to_text` to `from src.tools.html_text import _html_to_text`.
- [x] **Step 2:** `pytest tests/unit/test_search_tools.py` green.
- [x] **Verify:** grep confirms `_html_to_text`/`_SemanticTextParser` no longer defined in `search.py`; search.py LOC reduced (~676 → ~595).
