# Playwright-CLI Browser Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `browser.py` as a third web-search retrieval backend — a drop-in alternative to `google.py` (Google Custom Search API) and `serp.py` (SerpAPI) that requires no API key, using `playwright-cli` to drive a real browser against Google Search.

**Why:** All three backends share the same `POST /retrieve` contract. `browser.py` fills the gap when API keys aren't available or cost is a concern — at the tradeoff of higher per-query latency and exposure to bot detection.

| | `google.py` | `serp.py` | `browser.py` |
|---|---|---|---|
| API key required | Yes (`GOOGLE_API_KEY` + `GOOGLE_CSE_ID`) | Yes (`SERP_API_KEY`) | No |
| Rate limits | Quota-based | Paid plan | Browser fingerprint only |
| Result source | Google CSE (limited index) | Full Google SERP | Full Google SERP |
| Extra dependency | `google-api-python-client` | `requests` | `playwright-cli` (npm global) |
| Cost | Paid beyond free tier | Paid | Free (compute only) |
| Latency | ~1s | ~1s | ~5–10s (full browser round-trip) |

**Architecture:** `BrowserSearchEngine` shells out to the globally-installed `playwright-cli` binary in six sequential steps per query — `open`, `snapshot` (page load), `fill --submit`, `snapshot` (wait for SERP), `--raw eval` (JS DOM extraction), `close` — using named sessions so batch queries can run in parallel. Results are formatted via the shared `format_document` helper and served through the standard `create_search_app` FastAPI app.

**Tech Stack:** Python `subprocess`, `concurrent.futures.ThreadPoolExecutor`, `playwright-cli` CLI (global npm install `@playwright/cli`), FastAPI via existing `app.py` helpers.

**Fits existing pattern:** Mirrors `serp.py` / `google.py` — frozen dataclass config, `batch_search` → `_search_and_process`, `create_app(config)` helper, `parse_args()` + `main()`.

---

## Key corrections vs initial draft

| Issue | Original | Fixed |
|-------|----------|-------|
| `--raw` flag position | `playwright-cli -s=session --raw eval JS` | `playwright-cli --raw -s=session eval JS` |
| Missing `snapshot` after open | ❌ eval runs before page loads | ✅ snapshot waits for page to settle |
| Missing `snapshot` after fill | ❌ eval runs before SERP renders | ✅ snapshot waits for results |
| `create_app(config)` helper | `main()` calls `create_search_app` directly | Dedicated `create_app(config)` like serp.py |
| `batch_search` empty-queries guard | `min(len(queries), workers)` → 0-worker crash | `min(max(len(queries), 1), workers)` |
| Test mock call count | 4 per query | 6 per query (two snapshot calls added) |

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/internal/servers/retrieval/browser.py` | Engine + CLI entrypoint |
| Create | `tests/unit/retrieval/test_browser_retrieval.py` | Unit tests (subprocess mocked) |
| Modify | `.claude/skills/playwright-cli/SKILL.md` | Add browser-retrieval example |

---

### Task 1: Skeleton — `BrowserSearchEngine` with subprocess calls

**Files:**
- Create: `src/internal/servers/retrieval/browser.py`

- [ ] **Step 1: Write the failing test for single-query extraction**

```python
# tests/unit/retrieval/test_browser_retrieval.py
from __future__ import annotations
import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.internal.servers.retrieval.browser import BrowserSearchEngine, BrowserSearchConfig

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
    with patch("src.internal.servers.retrieval.browser.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _make_proc(),             # open
            _make_proc(),             # snapshot (page load)
            _make_proc(),             # fill --submit
            _make_proc(),             # snapshot (SERP results)
            _make_proc(FAKE_RESULTS), # --raw eval
            _make_proc(),             # close
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
# src/internal/servers/retrieval/browser.py
"""FastAPI retrieval server that uses playwright-cli for browser-based search."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from fastapi import FastAPI

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

# Extracts organic results from a Google SERP via h3 headings — more stable than class names.
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

    def _run(
        self,
        *args: str,
        session: str | None = None,
        raw: bool = False,
    ) -> subprocess.CompletedProcess:
        # --raw is a global flag; must precede -s= per playwright-cli CLI contract.
        cmd = [PLAYWRIGHT_CMD]
        if raw:
            cmd.append("--raw")
        if session:
            cmd.append(f"-s={session}")
        cmd.extend(args)
        return subprocess.run(
            cmd,
            capture_output=raw,
            text=True,
            timeout=self.config.subprocess_timeout,
        )

    def _search_and_process(self, query: str) -> list[dict[str, dict[str, str]]]:
        session = f"search-{uuid.uuid4().hex[:8]}"
        try:
            self._run("open", SEARCH_URL, "--persistent", session=session)
            self._run("snapshot", session=session)        # wait for page to settle; discover refs
            self._run(
                "fill",
                "getByRole('combobox', { name: 'Search' })",
                query,
                "--submit",
                session=session,
            )
            self._run("snapshot", session=session)        # wait for SERP results to render
            proc = self._run("eval", _EXTRACT_JS, session=session, raw=True)
            hits: list[dict] = json.loads(proc.stdout.strip()) if proc.stdout.strip() else []
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
        max_workers = min(max(len(queries), 1), self.config.batch_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._search_and_process, queries))


def create_app(config: BrowserSearchConfig) -> FastAPI:
    return create_search_app("Browser Retrieval (playwright-cli)", BrowserSearchEngine(config))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser-based retrieval server (playwright-cli)")
    add_host_port_args(parser, "BROWSER_RETRIEVAL_HOST", "BROWSER_RETRIEVAL_PORT", DEFAULT_HOST, DEFAULT_PORT)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    load_environment()
    args = parse_args()
    config = BrowserSearchConfig(topk=args.topk, batch_workers=args.workers)
    app = create_app(config)
    run_uvicorn_app(app, host=args.host, port=args.port)


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
git add src/internal/servers/retrieval/browser.py tests/unit/retrieval/test_browser_retrieval.py
git commit -m "feat: add BrowserSearchEngine using playwright-cli subprocess"
```

---

### Task 2: Edge-case tests — empty results, subprocess failure, topk truncation

**Files:**
- Modify: `tests/unit/retrieval/test_browser_retrieval.py`

- [ ] **Step 1: Write failing tests**

Each call sequence is 6 per query: open, snapshot, fill, snapshot, eval, close.

```python
# append to tests/unit/retrieval/test_browser_retrieval.py

def test_empty_results_when_eval_returns_empty_list():
    engine = BrowserSearchEngine(BrowserSearchConfig(topk=5))
    with patch("src.internal.servers.retrieval.browser.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _make_proc(),      # open
            _make_proc(),      # snapshot
            _make_proc(),      # fill
            _make_proc(),      # snapshot
            _make_proc("[]"),  # eval → empty list
            _make_proc(),      # close
        ]
        results = engine._search_and_process("obscure query xyz")
    assert results == []


def test_subprocess_timeout_returns_empty_and_closes():
    engine = BrowserSearchEngine(BrowserSearchConfig(topk=5))
    with patch("src.internal.servers.retrieval.browser.subprocess.run") as mock_run:
        # open succeeds, snapshot times out, close still called in finally
        mock_run.side_effect = [
            _make_proc(),
            subprocess.TimeoutExpired(cmd="playwright-cli", timeout=30),
            _make_proc(),  # close in finally block
        ]
        results = engine._search_and_process("query")
    assert results == []
    assert mock_run.call_count == 3  # open, snapshot (timeout), close


def test_topk_truncates_results():
    engine = BrowserSearchEngine(BrowserSearchConfig(topk=1))
    many = json.dumps([
        {"title": f"Result {i}", "url": f"https://example{i}.com", "snippet": "x"}
        for i in range(5)
    ])
    with patch("src.internal.servers.retrieval.browser.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _make_proc(), _make_proc(), _make_proc(), _make_proc(),
            _make_proc(many), _make_proc(),
        ]
        results = engine._search_and_process("query")
    assert len(results) == 1
    assert results[0]["document"]["title"] == "Result 0"


def test_batch_search_runs_queries_in_parallel():
    engine = BrowserSearchEngine(BrowserSearchConfig(topk=2, batch_workers=2))
    single = json.dumps([{"title": "T", "url": "https://t.com", "snippet": "s"}])
    with patch("src.internal.servers.retrieval.browser.subprocess.run") as mock_run:
        # 6 calls per query × 2 queries = 12 calls
        mock_run.side_effect = [
            _make_proc(), _make_proc(), _make_proc(), _make_proc(), _make_proc(single), _make_proc(),
            _make_proc(), _make_proc(), _make_proc(), _make_proc(), _make_proc(single), _make_proc(),
        ]
        results = engine.batch_search(["q1", "q2"])
    assert len(results) == 2
    assert results[0][0]["document"]["title"] == "T"
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
pytest tests/unit/retrieval/test_browser_retrieval.py -v
```

Expected: all 5 tests `PASSED`

- [ ] **Step 3: Commit**

```bash
git add tests/unit/retrieval/test_browser_retrieval.py
git commit -m "test: add edge-case coverage for BrowserSearchEngine"
```

---

### Task 3: Wire into module and verify server starts

**Files:**
- Check: `src/internal/servers/retrieval/__init__.py` (empty — no changes needed)

- [ ] **Step 1: Verify the module entry point is runnable**

```bash
python3 -m src.internal.servers.retrieval.browser --help
```

Expected output includes `--topk`, `--host`, `--port`, `--workers`.

- [ ] **Step 2: Smoke-test the server starts**

```bash
timeout 3 python3 -m src.internal.servers.retrieval.browser --port 8099 2>&1 || true
```

Expected: logs `Uvicorn running on ...`, killed by timeout. No import errors.

- [ ] **Step 3: Commit (only if `__init__.py` needed changes)**

If `__init__.py` was empty (it is), skip. Otherwise:

```bash
git add src/internal/servers/retrieval/__init__.py
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
python3 -m src.internal.servers.retrieval.browser --port 8099 &
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

---

### Task 5: Update playwright-cli skill with browser-retrieval pattern

**Files:**
- Modify: `.claude/skills/playwright-cli/SKILL.md`

- [ ] **Step 1: Append the browser-retrieval example to the skill**

Add the following section at the end of `.claude/skills/playwright-cli/SKILL.md`:

```markdown
## Example: Scripted search and result extraction

Drive a Google search and extract top results as JSON using playwright-cli subprocess calls
(as used by `src/internal/servers/retrieval/browser.py`):

```bash
SESSION="search-$(openssl rand -hex 4)"
playwright-cli -s=$SESSION open https://www.google.com --persistent
playwright-cli -s=$SESSION snapshot                          # wait for page to settle
playwright-cli -s=$SESSION fill "getByRole('combobox', { name: 'Search' })" "what is FAISS" --submit
playwright-cli -s=$SESSION snapshot                          # wait for SERP results to render
playwright-cli --raw -s=$SESSION eval \
  "JSON.stringify([...document.querySelectorAll('h3')].filter(h=>h.closest('a')).slice(0,5).map(h=>({title:h.textContent.trim(),url:h.closest('a').href,snippet:(h.closest('[data-hveid]')?.lastElementChild?.textContent?.trim()||'')})).filter(r=>r.url&&!r.url.includes('google.com/search')))"
playwright-cli -s=$SESSION close
```

Note: `--raw` is a global flag and must precede `-s=` in the command.
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
- [x] `snapshot` after `open` and after `fill --submit` (correct playwright-cli flow) → Task 1
- [x] `--raw` before `-s=` in cmd builder → Task 1
- [x] `create_app(config)` helper matching serp.py/google.py pattern → Task 1
- [x] `min(max(len(queries), 1), workers)` guard → Task 1
- [x] Follows `format_document` / `create_search_app` contract → Task 1
- [x] CLI entrypoint (`python3 -m ...`) → Task 3
- [x] Live validation → Task 4
- [x] Skill updated with correct command order → Task 5

**Type consistency:**
- `_search_and_process` returns `list[dict[str, dict[str, str]]]` ✓
- `format_document(title, snippet, url)` matches app.py signature ✓
- `BrowserSearchConfig` frozen dataclass, all fields have defaults ✓
- `create_app(config) -> FastAPI` matches serp.py/google.py pattern ✓
