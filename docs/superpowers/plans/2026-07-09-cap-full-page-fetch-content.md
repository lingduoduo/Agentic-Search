# Cap Full-Page Fetch Content Implementation Plan

> Use superpowers:test-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Cap per-page `<fetch>` content in `_format_full_page_information` (default 4096 chars, head-keep) so a large fetched page can't blow the token budget.

**Architecture:** All in `src/agents/search/search.py`: a config field, a pure `_truncate_page_content` helper, and one call in `_format_full_page_information`.

**Tech Stack:** Python, SearchAgentLoop.

## Global Constraints

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

```python
"""Unit tests for full-page fetch content truncation."""

from __future__ import annotations

from src.agents.search.search import (
    SearchAgentLoop,
    SearchAgentLoopConfig,
    _truncate_page_content,
)
from src.context.search import SearchResult
from tests.unit.test_agent_loop import (
    DummyServerManager,
    DummyTokenizerWithEncode,
)


def test_truncate_under_limit_unchanged():
    assert _truncate_page_content("short", 100) == "short"


def test_truncate_over_limit_head_kept_with_marker():
    text = "x" * 500
    out = _truncate_page_content(text, 100)
    assert out == "x" * 100 + "…(truncated)"
    assert out.startswith("x" * 100)
    assert out.endswith("…(truncated)")


def test_truncate_disabled_when_limit_non_positive():
    text = "y" * 500
    assert _truncate_page_content(text, 0) == text
    assert _truncate_page_content(text, -1) == text


def _loop(max_chars: int) -> SearchAgentLoop:
    return SearchAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
        search_config=SearchAgentLoopConfig(max_full_page_chars=max_chars),
    )


def test_format_full_page_truncates_large_page():
    loop = _loop(50)
    pages = [SearchResult(contents="z" * 500, title="Big", url="http://x")]
    block = loop._format_full_page_information(pages)
    assert "z" * 50 + "…(truncated)" in block
    assert "z" * 51 not in block  # content beyond the cap is gone


def test_format_full_page_keeps_small_page_intact():
    loop = _loop(4096)
    pages = [SearchResult(contents="small body", title="S", url="http://y")]
    block = loop._format_full_page_information(pages)
    assert "small body" in block
    assert "truncated" not in block


def test_default_cap_is_4096():
    assert SearchAgentLoopConfig().max_full_page_chars == 4096
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_full_page_truncation.py -v`
Expected: FAIL — `ImportError` for `_truncate_page_content` / no `max_full_page_chars`.

- [ ] **Step 3: Write minimal implementation**

In `src/agents/search/search.py`:

(a) Add the config field to `SearchAgentLoopConfig`, near `full_page_obs_template`:
```python
    # Max characters of a single fetched page's content inlined into the
    # <full_page> observation. <= 0 disables the cap. Guards the token budget
    # (full pages are otherwise inlined verbatim).
    max_full_page_chars: int = 4096
```

(b) Add a module-level helper (near other module-level helpers, e.g. beside `_result_fingerprint`):
```python
def _truncate_page_content(text: str, limit: int) -> str:
    """Head-keep a fetched page's content to `limit` chars (<= 0 disables)."""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "…(truncated)"
```

(c) In `_format_full_page_information`, replace `sections.append(page.contents)` with:
```python
            sections.append(
                _truncate_page_content(
                    page.contents, self.search_config.max_full_page_chars
                )
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_full_page_truncation.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Regression — existing agent-loop tests**

Run: `python3 -m pytest tests/unit/test_agent_loop.py -q`
Expected: PASS (default 4096 leaves existing small fixtures unaffected).

- [ ] **Step 6: Commit**

```bash
git add src/agents/search/search.py tests/unit/test_full_page_truncation.py
git commit -m "feat(search): cap full-page fetch content in the observation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

- Config field `max_full_page_chars=4096` (spec §Approach 1) → Task 1 Step 3a + `test_default_cap_is_4096`. ✓
- Pure helper `_truncate_page_content` (spec §Approach 2) → Step 3b + 3 helper tests. ✓
- Applied per page in `_format_full_page_information` (spec §Approach 3) → Step 3c + 2 formatter tests. ✓
- `<= 0` disables (spec success criteria) → `test_truncate_disabled_when_limit_non_positive`. ✓
- No-model constraint → pure-helper tests need nothing; formatter tests reuse existing dummies. ✓
- Types consistent: `_truncate_page_content(text: str, limit: int) -> str` identical in def, tests, and the formatter call. ✓
