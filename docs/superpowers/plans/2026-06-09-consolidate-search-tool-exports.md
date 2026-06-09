# Search Tool Export Consolidation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose `search_tool` and `fetch_pages_concurrently` through the `src/tools` package `__init__.py` and update the web backend to import from the package rather than directly from the module.

**Architecture:** `src/tools/__init__.py` is the public interface for the tools package. Two functions (`search_tool`, `fetch_pages_concurrently`) are used by `src/backend/servers/web/app.py` but imported directly from `src.tools.search`, bypassing the package API. Adding them to `__init__.py` and updating the backend import fixes the inconsistency without changing any behavior.

**Tech Stack:** Python 3.12, pytest

---

## File Map

| File | Change |
|---|---|
| `src/tools/__init__.py` | Add two re-exports: `search_tool`, `fetch_pages_concurrently` |
| `src/backend/servers/web/app.py` | Change `from src.tools.search import ...` → `from src.tools import ...` for these two names |

---

## Task 1: Export the missing functions from `src/tools/__init__.py`

**Files:**
- Modify: `src/tools/__init__.py`

- [ ] **Step 1: Run the existing tests to confirm baseline**

```bash
pytest tests/unit/test_search_tools.py tests/unit/test_api_tools.py -v --tb=short 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 2: Add the two missing re-exports to `src/tools/__init__.py`**

Current `src/tools/__init__.py` (lines 14–19):
```python
from .search import SearchPage as SearchPage
from .search import build_search_tool as build_search_tool
from .search import fetch_url as fetch_url
from .search import format_search_pages as format_search_pages
```

Replace with:
```python
from .search import SearchPage as SearchPage
from .search import build_search_tool as build_search_tool
from .search import fetch_pages_concurrently as fetch_pages_concurrently
from .search import fetch_url as fetch_url
from .search import format_search_pages as format_search_pages
from .search import search_tool as search_tool
```

- [ ] **Step 3: Run tests to verify nothing broke**

```bash
pytest tests/unit/test_search_tools.py tests/unit/test_api_tools.py -v --tb=short 2>&1 | tail -20
```

Expected: same pass count as Step 1.

- [ ] **Step 4: Commit**

```bash
git add src/tools/__init__.py
git commit -m "$(cat <<'EOF'
feat: export search_tool and fetch_pages_concurrently from src/tools

Both functions are part of the public API used by the web backend
but were missing from the package's __init__.py.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Update the web backend to import from the package, not the module

**Files:**
- Modify: `src/backend/servers/web/app.py`

- [ ] **Step 1: Update the three-line import block in `app.py`**

Find (around line 79–81):
```python
from src.tools.search import SearchPage
from src.tools.search import fetch_pages_concurrently
from src.tools.search import search_tool
```

Replace with:
```python
from src.tools import SearchPage
from src.tools import fetch_pages_concurrently
from src.tools import search_tool
```

- [ ] **Step 2: Run the full unit test suite**

```bash
pytest tests/unit/ -v --tb=short 2>&1 | tail -30
```

Expected: 1228 tests pass (same as baseline from previous session).

- [ ] **Step 3: Commit**

```bash
git add src/backend/servers/web/app.py
git commit -m "$(cat <<'EOF'
refactor: import search_tool and fetch_pages_concurrently via src.tools package

Use the package's public __init__.py interface instead of importing
directly from the sub-module.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```
