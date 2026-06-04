# Playwright-CLI Browser Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `browser.py` retrieval backend that uses `playwright-cli` subprocess calls to search Google and return results, fitting the existing `/retrieve` API contract.

**Architecture:** `BrowserSearchEngine` shells out to the globally-installed `playwright-cli` binary in three sequential steps per query — `open`, `fill --submit`, `eval` (JS DOM extraction) — using named sessions so batch queries can run in parallel. Results are formatted via the shared `format_document` helper and served through the standard `create_search_app` FastAPI app.

**Tech Stack:** Python `subprocess`, `concurrent.futures.ThreadPoolExecutor`, `playwright-cli` CLI (global npm install `@playwright/cli`), FastAPI via existing `app.py` helpers.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/backend/servers/retrieval/browser.py` | Engine + CLI entrypoint |
| Create | `tests/unit/retrieval/test_browser_retrieval.py` | Unit tests (subprocess mocked) |
| Modify | `.claude/skills/playwright-cli/SKILL.md` | Add browser-retrieval example |

---

### Task 1: Skeleton — `BrowserSearchEngine` with subprocess calls

**Files:**
- Create: `src/backend/servers/retrieval/browser.py`

- [ ] **Step 1: Write the failing test for single-query extraction**

```python
# tests/unit/retrieval/test_browser_retrieval.py
from __future__ import annotations
import json
from unittest.mock import MagicMock, call, patch

import pytest

from src.backend.servers.retrieval.browser import BrowserSearchEngine, BrowserSearchConfig

FAKE_RESULTS = json.dumps([
    {"title": "FAISS - Wikipedia", "url": "https://en.wikipedia.org/wiki/FAISS", "snippet": "A library for similarity search."},
    {"title": "GitHub facebookresearch/faiss", "url": "https://github.com/facebookresearch/faiss", "snippet": "Efficient similarity search."},
])


def _make_proc(stdout: str = "", returncode: int = 0):
    m = MagicMock()
    m.stdout = stdout
    m.returncode = returncode
    return m


def test_search_query_returns_formatted_documents():
    engine = BrowserSearchEngine(BrowserSearchConfig(topk=2))
    with patch("src.backend.servers.retrieval.browser.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _make_proc(),            # open
            _make_proc(),            # fill --submit
            _make_proc(FAKE_RESULTS),# eval (returns JSON)
            _make_proc(),            # close
        ]
        results = engine._search_and_process("what is FAISS")

    assert len(results) == 2
    assert results[0]["document"]["title"] == "FAISS - Wikipedia"
    assert results[0]["document"]["url"] == "https://en.wikipedia.org/wiki/FAISS"
    assert "FAISS - Wikipedia" in results[0]["document"]["contents"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/linghuang/Git/Agentic-Search
pytest tests/unit/retrieval/test_browser_retrieval.py::test_search_query_returns_formatted_documents -v
```

Expected: `ModuleNotFoundError` — `browser.py` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# src/backend/servers/retrieval/browser.py
"""FastAPI retrieval server that uses playwright-cli for browser-based search."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .app import (
    add_host_port_args,
    create_search_app,
    format_document,
    load_environment,
    run_uvicorn_app,
)

logger = logging.getLogger(__name__)

DEFAULT_TOPK = 5
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
PLAYWRIGHT_CMD = "playwright-cli"
SEARCH_URL = "https://www.google.com"
SUBPROCESS_TIMEOUT = 30

# JS that extracts top organic results from a Google SERP page.
# Uses h3 headings (result titles) as anchor points — more stable than CSS class names.
_EXTRACT_JS = (
    "JSON.stringify("
    "[...document.querySelectorAll('h3')]"
    ".filter(h=>h.closest('a'))"
    ".slice(0,10)"
    ".map(h=>({title:h.textContent.trim(),"
    "url:h.closest('a').href,"
    "snippet:(h.closest('[data-hveid]')?.lastElementChild?.textContent?.trim()||'')}))"
    ".filter(r=>r.url&&!r.url.includes('google.com/search'))"
    ")"
)


@dataclass(frozen=True)
class BrowserSearchConfig:
    topk: int = DEFAULT_TOPK
    batch_workers: int = 4
    subprocess_timeout: int = SUBPROCESS_TIMEOUT


class BrowserSearchEngine:
    def __init__(self, config: BrowserSearchConfig):
        self.config = config

    def _run(self, *args: str, session: str | None = None, capture: bool = False) -> subprocess.CompletedProcess:
        cmd = [PLAYWRIGHT_CMD]
        if session:
            cmd.append(f"-s={session}")
        cmd.extend(args)
        return subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=self.config.subprocess_timeout,
        )

    def _search_and_process(self, query: str) -> list[dict[str, dict[str, str]]]:
        session = f"search-{uuid.uuid4().hex[:8]}"
        try:
            self._run("open", SEARCH_URL, "--persistent", session=session)
            self._run(
                "fill",
                "getByRole('combobox', { name: 'Search' })",
                query,
                "--submit",
                session=session,
            )
            proc = self._run("--raw", "eval", _EXTRACT_JS, session=session, capture=True)
            raw = proc.stdout.strip()
            hits: list[dict] = json.loads(raw) if raw else []
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as exc:
            logger.warning("browser search failed for %r: %s", query, exc)
            hits = []
        finally:
            try:
                self._run("close", session=session)
            except Exception:
                pass

        return [
            format_document(h.get("title"), h.get("snippet"), h.get("url"))
            for h in hits[: self.config.topk]
        ]

    def batch_search(self, queries: list[str]) -> list[list[dict[str, dict[str, str]]]]:
        max_workers = min(len(queries), self.config.batch_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._search_and_process, queries))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Browser-based retrieval server (playwright-cli)")
    add_host_port_args(parser, "BROWSER_RETRIEVAL_HOST", "BROWSER_RETRIEVAL_PORT", DEFAULT_HOST, DEFAULT_PORT)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> None:
    load_environment()
    args = _build_arg_parser().parse_args()
    config = BrowserSearchConfig(topk=args.topk, batch_workers=args.workers)
    engine = BrowserSearchEngine(config)
    app = create_search_app("Browser Retrieval (playwright-cli)", engine)
    run_uvicorn_app(app, args.host, args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/retrieval/test_browser_retrieval.py::test_search_query_returns_formatted_documents -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/backend/servers/retrieval/browser.py tests/unit/retrieval/test_browser_retrieval.py
git commit -m "feat: add BrowserSearchEngine using playwright-cli subprocess"
```

---

### Task 2: Edge-case tests — empty results, subprocess failure, topk truncation

**Files:**
- Modify: `tests/unit/retrieval/test_browser_retrieval.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/unit/retrieval/test_browser_retrieval.py

def test_empty_results_when_eval_returns_empty_list():
    engine = BrowserSearchEngine(BrowserSearchConfig(topk=5))
    with patch("src.backend.servers.retrieval.browser.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _make_proc(),       # open
            _make_proc(),       # fill
            _make_proc("[]"),   # eval → empty list
            _make_proc(),       # close
        ]
        results = engine._search_and_process("obscure query xyz")
    assert results == []


def test_subprocess_timeout_returns_empty_and_closes():
    engine = BrowserSearchEngine(BrowserSearchConfig(topk=5))
    with patch("src.backend.servers.retrieval.browser.subprocess.run") as mock_run:
        # open succeeds, fill times out, close still called
        mock_run.side_effect = [
            _make_proc(),
            subprocess.TimeoutExpired(cmd="playwright-cli", timeout=30),
            _make_proc(),  # close in finally block
        ]
        results = engine._search_and_process("query")
    assert results == []
    assert mock_run.call_count == 3  # open, fill (timeout), close


def test_topk_truncates_results():
    engine = BrowserSearchEngine(BrowserSearchConfig(topk=1))
    many = json.dumps([
        {"title": f"Result {i}", "url": f"https://example{i}.com", "snippet": "x"}
        for i in range(5)
    ])
    with patch("src.backend.servers.retrieval.browser.subprocess.run") as mock_run:
        mock_run.side_effect = [_make_proc(), _make_proc(), _make_proc(many), _make_proc()]
        results = engine._search_and_process("query")
    assert len(results) == 1
    assert results[0]["document"]["title"] == "Result 0"


def test_batch_search_runs_queries_in_parallel():
    engine = BrowserSearchEngine(BrowserSearchConfig(topk=2, batch_workers=2))
    single = json.dumps([{"title": "T", "url": "https://t.com", "snippet": "s"}])
    with patch("src.backend.servers.retrieval.browser.subprocess.run") as mock_run:
        # 4 calls per query × 2 queries = 8 calls
        mock_run.side_effect = [
            _make_proc(), _make_proc(), _make_proc(single), _make_proc(),
            _make_proc(), _make_proc(), _make_proc(single), _make_proc(),
        ]
        results = engine.batch_search(["q1", "q2"])
    assert len(results) == 2
    assert results[0][0]["document"]["title"] == "T"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/retrieval/test_browser_retrieval.py -v -k "empty or timeout or topk or batch"
```

Expected: `ImportError` on `subprocess.TimeoutExpired` in test file — add `import subprocess` to the test file.

- [ ] **Step 3: Add missing import to test file**

```python
# at top of tests/unit/retrieval/test_browser_retrieval.py, add:
import subprocess
```

- [ ] **Step 4: Run tests again**

```bash
pytest tests/unit/retrieval/test_browser_retrieval.py -v
```

Expected: all 5 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add tests/unit/retrieval/test_browser_retrieval.py
git commit -m "test: add edge-case coverage for BrowserSearchEngine"
```

---

### Task 3: Wire into module `__init__` and verify server starts

**Files:**
- Modify: `src/backend/servers/retrieval/__init__.py`

- [ ] **Step 1: Check current `__init__.py`**

```bash
cat src/backend/servers/retrieval/__init__.py
```

- [ ] **Step 2: Add browser to `__init__.py` exports (only if the file exports other engines)**

If `__init__.py` is empty, skip this step. If it imports other engines, add:

```python
from .browser import BrowserSearchConfig, BrowserSearchEngine
```

- [ ] **Step 3: Verify the module entry point is runnable**

```bash
python3 -m src.backend.servers.retrieval.browser --help
```

Expected output includes `--topk`, `--host`, `--port`, `--workers`.

- [ ] **Step 4: Smoke-test the server starts (no live browser needed)**

```bash
timeout 3 python3 -m src.backend.servers.retrieval.browser --port 8099 2>&1 || true
```

Expected: server starts, logs `Uvicorn running on ...`, then killed by timeout. No import errors.

- [ ] **Step 5: Commit**

```bash
git add src/backend/servers/retrieval/__init__.py
git commit -m "feat: expose BrowserSearchEngine in retrieval package"
```

---

### Task 4: Live smoke test with real playwright-cli

**Files:** none (manual validation step)

- [ ] **Step 1: Kill any stale playwright sessions**

```bash
playwright-cli kill-all 2>/dev/null || true
```

- [ ] **Step 2: Start the server**

```bash
python3 -m src.backend.servers.retrieval.browser --port 8099 &
SERVER_PID=$!
sleep 2
```

- [ ] **Step 3: Send a real search request**

```bash
curl -s -X POST http://localhost:8099/retrieve \
  -H "Content-Type: application/json" \
  -d '{"queries": ["what is FAISS"]}' | python3 -m json.tool
```

Expected: JSON with `result` array containing documents with `title`, `contents`, `url` fields.

- [ ] **Step 4: Kill the server**

```bash
kill $SERVER_PID 2>/dev/null || true
playwright-cli kill-all 2>/dev/null || true
```

- [ ] **Step 5: Commit nothing (validation-only task)**

---

### Task 5: Update playwright-cli skill with browser-retrieval pattern

**Files:**
- Modify: `.claude/skills/playwright-cli/SKILL.md`

- [ ] **Step 1: Append the browser-retrieval example to the skill**

Add the following section at the end of `.claude/skills/playwright-cli/SKILL.md`:

```markdown
## Example: Scripted search and result extraction

Drive a Google search and extract top results as JSON using playwright-cli subprocess calls
(as used by `src/backend/servers/retrieval/browser.py`):

```bash
SESSION="search-$(openssl rand -hex 4)"
playwright-cli -s=$SESSION open https://www.google.com --persistent
playwright-cli -s=$SESSION fill "getByRole('combobox', { name: 'Search' })" "what is FAISS" --submit
playwright-cli -s=$SESSION --raw eval \
  "JSON.stringify([...document.querySelectorAll('h3')].filter(h=>h.closest('a')).slice(0,5).map(h=>({title:h.textContent.trim(),url:h.closest('a').href,snippet:(h.closest('[data-hveid]')?.lastElementChild?.textContent?.trim()||'')})).filter(r=>r.url&&!r.url.includes('google.com/search')))"
playwright-cli -s=$SESSION close
```
```

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/playwright-cli/SKILL.md
git commit -m "docs: add browser-retrieval example to playwright-cli skill"
```

---

## Self-Review

**Spec coverage:**
- [x] `BrowserSearchEngine.batch_search` → Tasks 1–2
- [x] Per-query named sessions for parallel execution → Task 1
- [x] Follows `format_document` / `create_search_app` contract → Task 1
- [x] CLI entrypoint (`python3 -m ...`) → Task 3
- [x] Live validation → Task 4
- [x] Skill updated → Task 5

**Placeholder scan:** No TBDs. All code blocks are complete.

**Type consistency:**
- `_search_and_process` returns `list[dict[str, dict[str, str]]]` (matches `batch_search` inner type) ✓
- `format_document` signature: `(title, content, url)` → used as `format_document(h["title"], h["snippet"], h["url"])` ✓
- `BrowserSearchConfig` frozen dataclass — all fields have defaults ✓
