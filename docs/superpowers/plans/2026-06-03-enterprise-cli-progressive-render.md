# Enterprise Knowledge CLI — Progressive Markdown Rendering

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/cli/query.py` — a CLI that authenticates with a personal access token + user ID, queries the web backend's `/api/agent` endpoint, and progressively reveals the answer as animated markdown using `rich.live`.

**Architecture:** Three focused modules under `src/cli/` — `_auth.py` resolves the JWT (pre-baked or minted via `generate_user_jwt_token`), `_client.py` sends the async `httpx` call to `/api/agent`, and `_render.py` streams the answer word-by-word via `rich.live` + `rich.markdown.Markdown`. `query.py` is the argparse entry point that wires them together. No backend changes required.

**Tech Stack:** Python 3.11+, `httpx` 0.28 (already installed), `rich` 15 (already installed), `pytest` + `unittest.mock`.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/cli/__init__.py` | Package marker |
| Create | `src/cli/_auth.py` | `resolve_token()` — pre-baked JWT or mint via `generate_user_jwt_token` |
| Create | `src/cli/_client.py` | `query_agent()` — async `httpx` POST to `/api/agent`, returns `AgentResult` |
| Create | `src/cli/_render.py` | `render_sources()` + `render_answer_progressive()` using `rich.live` |
| Create | `src/cli/query.py` | `main()` argparse entry point, `__main__` block |
| Create | `tests/unit/test_cli_auth.py` | Unit tests for `_auth.py` |
| Create | `tests/unit/test_cli_client.py` | Unit tests for `_client.py` (mocked `httpx`) |
| Create | `tests/unit/test_cli_render.py` | Unit tests for `_render.py` (mocked `rich.live`) |

---

### Task 1: `_auth.py` — Token resolution

**Files:**
- Create: `src/cli/__init__.py`
- Create: `src/cli/_auth.py`
- Test: `tests/unit/test_cli_auth.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_cli_auth.py
from __future__ import annotations

import pytest

from src.cli._auth import resolve_token


def test_pre_baked_token_returned_as_is():
    token = resolve_token(token="my.jwt.here", user_id=None, email=None, secret=None)
    assert token == "my.jwt.here"


def test_token_takes_priority_over_user_id():
    token = resolve_token(token="pre.baked", user_id="alice", email=None, secret=None)
    assert token == "pre.baked"


def test_user_id_mints_jwt(monkeypatch):
    minted = "minted.jwt"
    monkeypatch.setattr(
        "src.cli._auth.generate_user_jwt_token",
        lambda **_: minted,
    )
    result = resolve_token(
        token=None, user_id="alice", email="a@corp.com", secret="test-signing-secret"
    )
    assert result == minted


def test_no_token_no_user_id_raises():
    with pytest.raises(ValueError, match="--token or --user-id"):
        resolve_token(token=None, user_id=None, email=None, secret=None)
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/unit/test_cli_auth.py -v
```
Expected: `ModuleNotFoundError` — `src/cli/_auth` does not exist yet.

- [ ] **Step 3: Create package marker**

```python
# src/cli/__init__.py
```
(empty file)

- [ ] **Step 4: Write `_auth.py`**

```python
# src/cli/_auth.py
from __future__ import annotations

from src.backend.auth import generate_user_jwt_token


def resolve_token(
    token: str | None,
    user_id: str | None,
    email: str | None = None,
    secret: str | None = None,
) -> str:
    """Return a Bearer JWT.

    Priority: pre-baked ``token`` > mint from ``user_id``.
    Raises ``ValueError`` if neither is supplied.
    """
    if token:
        return token
    if user_id:
        return generate_user_jwt_token(user_id=user_id, email=email, secret=secret)
    raise ValueError("Provide --token or --user-id to authenticate.")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/test_cli_auth.py -v
```
Expected: 4 tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add src/cli/__init__.py src/cli/_auth.py tests/unit/test_cli_auth.py
git commit -m "feat: add cli _auth module — resolve_token from pre-baked JWT or user_id"
```

---

### Task 2: `_client.py` — Async httpx call to `/api/agent`

**Files:**
- Create: `src/cli/_client.py`
- Test: `tests/unit/test_cli_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_cli_client.py
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cli._client import AgentResult, query_agent

FAKE_RESPONSE = {
    "session_id": "sess-123",
    "answer": "The quarterly report shows 12% revenue growth.",
    "citations": ["[1]"],
    "documents": [
        {
            "id": "doc1",
            "citation": "[1]",
            "title": "Q3 Financial Report",
            "content": "Revenue grew 12% year-over-year.",
            "url": "https://internal.corp/reports/q3",
            "score": 0.95,
            "metadata": {},
        }
    ],
    "messages": [],
    "hook_metadata": {},
}


def _make_mock_response(data: dict, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_query_agent_returns_agent_result():
    mock_resp = _make_mock_response(FAKE_RESPONSE)
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("src.cli._client.httpx.AsyncClient", return_value=mock_client):
        result = await query_agent(
            "http://localhost:7860", "show me the Q3 report", "my.token", top_k=5
        )

    assert isinstance(result, AgentResult)
    assert result.session_id == "sess-123"
    assert result.answer == "The quarterly report shows 12% revenue growth."
    assert len(result.documents) == 1
    assert result.documents[0]["title"] == "Q3 Financial Report"


@pytest.mark.asyncio
async def test_query_agent_sends_correct_headers_and_body():
    mock_resp = _make_mock_response(FAKE_RESPONSE)
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("src.cli._client.httpx.AsyncClient", return_value=mock_client):
        await query_agent("http://localhost:7860", "q", "tok123", top_k=3, session_id="s1")

    call_kwargs = mock_client.post.call_args
    assert call_kwargs.kwargs["headers"] == {"Authorization": "Bearer tok123"}
    body = call_kwargs.kwargs["json"]
    assert body["query"] == "q"
    assert body["top_k"] == 3
    assert body["session_id"] == "s1"


@pytest.mark.asyncio
async def test_query_agent_raises_on_http_error():
    mock_resp = _make_mock_response({}, status_code=401)
    mock_resp.raise_for_status.side_effect = Exception("401 Unauthorized")
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("src.cli._client.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(Exception, match="401"):
            await query_agent("http://localhost:7860", "q", "bad.token")
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/unit/test_cli_client.py -v
```
Expected: `ModuleNotFoundError` — `src/cli/_client` does not exist.

- [ ] **Step 3: Write `_client.py`**

```python
# src/cli/_client.py
from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class AgentResult:
    session_id: str
    answer: str
    citations: list[str]
    documents: list[dict]


async def query_agent(
    base_url: str,
    query: str,
    token: str,
    *,
    top_k: int = 5,
    session_id: str | None = None,
) -> AgentResult:
    """POST /api/agent and return a typed result.

    Raises ``httpx.HTTPStatusError`` on 4xx/5xx.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/api/agent",
            json={"query": query, "top_k": top_k, "session_id": session_id},
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        data = resp.json()

    return AgentResult(
        session_id=data["session_id"],
        answer=data["answer"],
        citations=data.get("citations", []),
        documents=data.get("documents", []),
    )
```

- [ ] **Step 4: Install `pytest-asyncio` if needed**

```bash
pip show pytest-asyncio 2>/dev/null || pip install pytest-asyncio
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/test_cli_client.py -v
```
Expected: 3 tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add src/cli/_client.py tests/unit/test_cli_client.py
git commit -m "feat: add cli _client module — async query_agent over /api/agent"
```

---

### Task 3: `_render.py` — Sources table + progressive markdown animation

**Files:**
- Create: `src/cli/_render.py`
- Test: `tests/unit/test_cli_render.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_cli_render.py
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

from rich.console import Console

from src.cli._render import render_answer_progressive, render_sources

DOCS = [
    {"citation": "[1]", "title": "Q3 Financial Report", "url": "https://internal.corp/reports/q3", "content": "x"},
    {"citation": "[2]", "title": None, "url": None, "content": "y"},
]


def test_render_sources_prints_table(capsys):
    buf = io.StringIO()
    test_console = Console(file=buf, highlight=False)
    with patch("src.cli._render.console", test_console):
        render_sources(DOCS)
    output = buf.getvalue()
    assert "Q3 Financial Report" in output
    assert "internal.corp" in output
    assert "[1]" in output


def test_render_sources_empty_list_prints_nothing(capsys):
    buf = io.StringIO()
    test_console = Console(file=buf, highlight=False)
    with patch("src.cli._render.console", test_console):
        render_sources([])
    assert buf.getvalue() == ""


def test_render_answer_progressive_calls_live_update():
    live_mock = MagicMock()
    live_mock.__enter__ = MagicMock(return_value=live_mock)
    live_mock.__exit__ = MagicMock(return_value=False)

    with patch("src.cli._render.Live", return_value=live_mock), \
         patch("src.cli._render.time.sleep"):
        render_answer_progressive("Hello world test", words_per_second=1000.0)

    # Live.update called once per word (3 words)
    assert live_mock.update.call_count == 3


def test_render_answer_progressive_empty_string():
    live_mock = MagicMock()
    live_mock.__enter__ = MagicMock(return_value=live_mock)
    live_mock.__exit__ = MagicMock(return_value=False)

    with patch("src.cli._render.Live", return_value=live_mock), \
         patch("src.cli._render.time.sleep"):
        render_answer_progressive("", words_per_second=1000.0)

    assert live_mock.update.call_count == 0
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/unit/test_cli_render.py -v
```
Expected: `ModuleNotFoundError` — `src/cli/_render` does not exist.

- [ ] **Step 3: Write `_render.py`**

```python
# src/cli/_render.py
from __future__ import annotations

import time

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.table import Table

console = Console()


def render_sources(documents: list[dict]) -> None:
    """Print a source table. No-op if documents is empty."""
    if not documents:
        return
    table = Table(title="Sources", show_header=True, header_style="bold cyan", box=None)
    table.add_column("Cite", style="dim", width=5)
    table.add_column("Title")
    table.add_column("URL", style="blue")
    for doc in documents:
        table.add_row(
            doc.get("citation") or "",
            doc.get("title") or "—",
            doc.get("url") or "—",
        )
    console.print(table)
    console.print()


def render_answer_progressive(
    answer: str,
    *,
    words_per_second: float = 30.0,
) -> None:
    """Reveal *answer* word-by-word as animated rich Markdown.

    ``words_per_second`` controls animation speed (default: 30 ≈ fast read pace).
    """
    words = answer.split()
    if not words:
        return

    delay = 1.0 / max(words_per_second, 1.0)
    accumulated = ""
    with Live(Markdown(""), console=console, refresh_per_second=20) as live:
        for word in words:
            accumulated += ("" if not accumulated else " ") + word
            live.update(Markdown(accumulated))
            time.sleep(delay)
    console.print()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_cli_render.py -v
```
Expected: 4 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/cli/_render.py tests/unit/test_cli_render.py
git commit -m "feat: add cli _render module — sources table + progressive markdown animation"
```

---

### Task 4: `query.py` — Argparse entry point

**Files:**
- Create: `src/cli/query.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_cli_query.py
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cli._client import AgentResult
from src.cli.query import main

RESULT = AgentResult(
    session_id="s1",
    answer="Revenue grew 12% year-over-year.",
    citations=["[1]"],
    documents=[{"citation": "[1]", "title": "Q3 Report", "url": "https://internal.corp/q3", "content": "c"}],
)


def test_main_returns_0_on_success():
    with patch("src.cli.query.resolve_token", return_value="tok"), \
         patch("src.cli.query.asyncio.run", return_value=RESULT), \
         patch("src.cli.query.render_sources"), \
         patch("src.cli.query.render_answer_progressive"), \
         patch("src.cli.query.console"):
        code = main(["show me the Q3 report", "--token", "tok", "--url", "http://x"])
    assert code == 0


def test_main_returns_1_on_auth_error():
    with patch("src.cli.query.resolve_token", side_effect=ValueError("no auth")), \
         patch("src.cli.query.console"):
        code = main(["q", "--url", "http://x"])
    assert code == 1


def test_main_returns_1_on_request_error():
    with patch("src.cli.query.resolve_token", return_value="tok"), \
         patch("src.cli.query.asyncio.run", side_effect=Exception("connection refused")), \
         patch("src.cli.query.console"):
        code = main(["q", "--token", "tok", "--url", "http://x"])
    assert code == 1


def test_main_returns_1_when_query_empty():
    with patch("src.cli.query.console") as mock_console:
        mock_console.input.return_value = ""
        code = main(["--token", "tok", "--url", "http://x"])
    assert code == 1
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/unit/test_cli_query.py -v
```
Expected: `ModuleNotFoundError` — `src/cli/query` does not exist.

- [ ] **Step 3: Write `query.py`**

```python
# src/cli/query.py
"""Enterprise knowledge CLI.

Usage:
    # pre-baked token
    python3 -m src.cli.query "summarise last quarter's results" \\
        --token <jwt> --url http://localhost:7860

    # mint token from credentials
    python3 -m src.cli.query "what is our refund policy?" \\
        --user-id alice --email alice@corp.com --secret "$AUTH_SECRET"

    # interactive prompt
    python3 -m src.cli.query --token <jwt>
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from src.cli._auth import resolve_token
from src.cli._client import query_agent
from src.cli._render import console, render_answer_progressive, render_sources


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m src.cli.query",
        description="Query enterprise knowledge from the command line.",
    )
    p.add_argument("query", nargs="?", help="Search query (prompted interactively if omitted)")
    p.add_argument("--url", default="http://localhost:7860", metavar="URL",
                   help="Web backend base URL (default: http://localhost:7860)")
    p.add_argument("--token", metavar="JWT",
                   help="Pre-generated personal access token / JWT")
    p.add_argument("--user-id", dest="user_id", metavar="ID",
                   help="Personal user ID — used to mint a JWT when --token is absent")
    p.add_argument("--email", metavar="EMAIL",
                   help="Email embedded in the minted JWT")
    p.add_argument("--secret", metavar="SECRET",
                   help="JWT signing secret (falls back to AUTH_SECRET env var)")
    p.add_argument("--top-k", dest="top_k", type=int, default=5, metavar="N",
                   help="Number of documents to retrieve (default: 5)")
    p.add_argument("--session-id", dest="session_id", metavar="ID",
                   help="Resume a prior chat session")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    query = args.query or console.input("[bold]Query:[/bold] ").strip()
    if not query:
        console.print("[red]No query provided.[/red]")
        return 1

    try:
        token = resolve_token(args.token, args.user_id, args.email, args.secret)
    except ValueError as exc:
        console.print(f"[red]Auth error:[/red] {exc}")
        return 1

    with console.status("[bold green]Searching enterprise knowledge…", spinner="dots"):
        try:
            result = asyncio.run(
                query_agent(
                    args.url, query, token,
                    top_k=args.top_k,
                    session_id=args.session_id,
                )
            )
        except Exception as exc:
            console.print(f"[red]Request failed:[/red] {exc}")
            return 1

    render_sources(result.documents)
    console.rule("[bold]Answer")
    render_answer_progressive(result.answer)
    console.print(f"[dim]session_id: {result.session_id}[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run all CLI tests**

```bash
pytest tests/unit/test_cli_auth.py tests/unit/test_cli_client.py \
       tests/unit/test_cli_render.py tests/unit/test_cli_query.py -v
```
Expected: all tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/cli/query.py tests/unit/test_cli_query.py
git commit -m "feat: add cli entry point — enterprise knowledge query with progressive markdown"
```

---

### Task 5: Full-suite check + live smoke test

**Files:** none (validation only)

- [ ] **Step 1: Run the full test suite**

```bash
pytest --tb=short -q
```
Expected: no new failures. If failures exist in pre-existing tests, investigate before proceeding.

- [ ] **Step 2: Start the local stack**

```bash
# Terminal 1 — retrieval server
python3 -m src.backend.servers.retrieval.demo --corpus_path data/corpus.jsonl &

# Terminal 2 — web backend
uvicorn src.backend.servers.web.app:app --host 127.0.0.1 --port 7860 &

sleep 3
curl -s http://localhost:7860/health
```
Expected: `{"status":"ok"}`

- [ ] **Step 3: Mint a token manually**

```bash
python3 - <<'EOF'
from src.backend.auth import generate_user_jwt_token
print(generate_user_jwt_token(user_id="dev-user", email="dev@local"))
EOF
```
Copy the printed token — use it as `<TOKEN>` below.

- [ ] **Step 4: Run the CLI**

```bash
python3 -m src.cli.query "summarise last quarter's results" \
  --token <TOKEN> --url http://localhost:7860
```
Expected output:
```
Sources
──────────────────────────────────────────────────────────────
 Cite  Title             URL
 [1]   …                 …
 [2]   …

──────────────── Answer ──────────────────────────────────────
According to the Q3 report… [text streams word by word]

session_id: <uuid>
```

- [ ] **Step 5: Test interactive prompt (no query arg)**

```bash
python3 -m src.cli.query --token <TOKEN>
# type any enterprise query at the prompt, e.g. "what is our refund policy?"
```
Expected: same output as above.

- [ ] **Step 6: Kill background servers**

```bash
pkill -f "retrieval.demo" 2>/dev/null; pkill -f "uvicorn.*app:app" 2>/dev/null; true
```

- [ ] **Step 7: Commit nothing** — validation-only task.

---

## Self-Review

**Spec coverage:**
- [x] CLI entry point (`python3 -m src.cli.query`) → Task 4
- [x] Personal access token (`--token`) → Tasks 1, 4
- [x] Personal ID mint path (`--user-id`, `--email`, `--secret`) → Tasks 1, 4
- [x] Progressive markdown rendering (word-by-word via `rich.live`) → Task 3
- [x] Sources table displayed before answer → Tasks 3, 4
- [x] Session resumption (`--session-id`) → Tasks 2, 4
- [x] Live smoke test → Task 5

**Placeholder scan:** None found — all code blocks are complete.

**Type consistency:**
- `resolve_token(token, user_id, email, secret) -> str` — matches usage in `query.py` ✓
- `query_agent(base_url, query, token, *, top_k, session_id) -> AgentResult` — matches `asyncio.run(query_agent(...))` call ✓
- `AgentResult.documents: list[dict]` — matches `render_sources(result.documents)` and `SourceDocumentView` keys (`citation`, `title`, `url`) ✓
- `render_sources(documents: list[dict])` / `render_answer_progressive(answer: str, *, words_per_second)` — match all call sites ✓
