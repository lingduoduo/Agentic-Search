# Public Data-Source Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the `/tools` Tool Agent nine keyless public data-source tools —
Wikipedia, ArXiv, Wayback, weather, stock quotes, crypto prices, currency
conversion, geocoding, and nearby places — and let the citeable ones render as
source cards.

**Architecture:** A new `src/internal/tools/public_data/` subpackage. One
`_http.py` module owns every outbound call (aiohttp session, User-Agent,
timeout, error translation) plus a `guarded` decorator that serializes results
and turns failures into `{"error": ...}`. Three theme modules each expose three
`build_*_tool()` factories returning `FunctionTool`s. `public_data_tools()`
collects the nine, and `tool_knowledge_base()` seeds them so they are
agent-callable by default.

**Tech Stack:** Python 3, aiohttp (via the existing lazy shim), stdlib
`xml.etree.ElementTree` and `json`, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-08-06-public-data-tools-design.md`

## Global Constraints

- **No new dependencies.** Nothing may be added to `requirements.txt`. ArXiv's
  Atom XML is parsed with stdlib `xml.etree.ElementTree`; do not install the
  `arxiv` or `wikipedia` packages.
- **No live network in tests.** Every test monkeypatches the `_http` layer.
- **Async only.** Never use `requests` — it blocks the event loop. All HTTP goes
  through `public_data._http`.
- **Error contract.** A tool never raises. Failures return the JSON string
  `{"error": "<message>"}`.
- **Citeable return contract.** `search_wikipedia`, `search_arxiv`, and
  `search_wayback` return a JSON **array** of `{"title", "content", "url"}`
  objects. Every other tool returns a JSON **object** of facts.
- **All nine declare** `effect=ToolEffect.READ_ONLY`.
- **No new environment variables and no feature flag.**
- Style: `from __future__ import annotations` at the top of every new module;
  module-level `logger = logging.getLogger(__name__)`; log at `debug`, never
  with emoji or `traceback.format_exc()`.
- Lint gate for every commit: `ruff check . --fix && ruff format .`

---

### Task 1: HTTP layer and package skeleton

**Files:**
- Create: `src/internal/tools/public_data/__init__.py`
- Create: `src/internal/tools/public_data/_http.py`
- Test: `tests/unit/test_public_data_http.py`

**Interfaces:**
- Consumes: `src.context.retrieval.client.aiohttp` (an existing lazy-import
  shim whose attributes stay monkeypatchable).
- Produces, used by Tasks 2–4:
  - `PublicDataError(Exception)`
  - `async get_json(url: str, *, params: dict | None = None, headers: dict | None = None, timeout_seconds: float = 10.0) -> Any`
  - `async get_text(url: str, *, params: dict | None = None, headers: dict | None = None, timeout_seconds: float = 10.0) -> str`
  - `async post_json(url: str, *, data: dict | str, headers: dict | None = None, timeout_seconds: float = 10.0) -> Any`
  - `guarded(fn: Callable) -> Callable` — wraps an async function returning a
    `dict`/`list` into one returning a JSON string, converting any exception
    into `{"error": ...}`.
  - `USER_AGENT: str`, `MAX_CONTENT_CHARS: int`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_public_data_http.py`:

```python
"""Unit tests for the public_data HTTP layer. No live network."""

from __future__ import annotations

import asyncio
import json

import pytest

from src.internal.tools.public_data import _http
from src.internal.tools.public_data._http import (
    PublicDataError,
    get_json,
    guarded,
)


class _FakeResponse:
    def __init__(self, *, status=200, body="{}"):
        self.status = status
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return self._body


class _FakeSession:
    """Records the single request it is given, then replays a canned response."""

    calls: list[dict] = []

    def __init__(self, *, status=200, body="{}", raises=None):
        self._status = status
        self._body = body
        self._raises = raises

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def request(self, method, url, **kwargs):
        if self._raises is not None:
            raise self._raises
        _FakeSession.calls.append({"method": method, "url": url, **kwargs})
        return _FakeResponse(status=self._status, body=self._body)


def _install(monkeypatch, **kwargs):
    _FakeSession.calls = []

    class _Aiohttp:
        @staticmethod
        def ClientTimeout(total=None):
            return total

        @staticmethod
        def ClientSession(timeout=None):
            return _FakeSession(**kwargs)

    monkeypatch.setattr(_http, "aiohttp", _Aiohttp)


def test_get_json_parses_body_and_sends_user_agent(monkeypatch):
    _install(monkeypatch, body=json.dumps({"ok": True}))

    result = asyncio.run(get_json("https://example.test/x", params={"a": "b"}))

    assert result == {"ok": True}
    call = _FakeSession.calls[0]
    assert call["method"] == "GET"
    assert call["params"] == {"a": "b"}
    assert call["headers"]["User-Agent"] == _http.USER_AGENT


def test_get_json_caller_headers_override_default(monkeypatch):
    _install(monkeypatch, body="{}")

    asyncio.run(get_json("https://example.test/x", headers={"User-Agent": "Mozilla/5.0"}))

    assert _FakeSession.calls[0]["headers"]["User-Agent"] == "Mozilla/5.0"


def test_get_json_raises_on_http_error(monkeypatch):
    _install(monkeypatch, status=503, body="down")

    with pytest.raises(PublicDataError) as excinfo:
        asyncio.run(get_json("https://example.test/x"))

    assert "503" in str(excinfo.value)


def test_get_json_raises_on_non_json_body(monkeypatch):
    _install(monkeypatch, body="<html>nope</html>")

    with pytest.raises(PublicDataError):
        asyncio.run(get_json("https://example.test/x"))


def test_get_json_raises_on_transport_failure(monkeypatch):
    _install(monkeypatch, raises=asyncio.TimeoutError())

    with pytest.raises(PublicDataError):
        asyncio.run(get_json("https://example.test/x"))


def test_guarded_serializes_success():
    @guarded
    async def _ok(value: str):
        return {"value": value}

    assert json.loads(asyncio.run(_ok(value="hi"))) == {"value": "hi"}


def test_guarded_converts_public_data_error():
    @guarded
    async def _boom():
        raise PublicDataError("upstream is down")

    assert json.loads(asyncio.run(_boom())) == {"error": "upstream is down"}


def test_guarded_converts_unexpected_error():
    @guarded
    async def _boom():
        raise KeyError("missing")

    assert "error" in json.loads(asyncio.run(_boom()))


def test_guarded_result_is_a_coroutine_function():
    """FunctionTool.execute awaits only if iscoroutinefunction() is True."""
    import inspect

    @guarded
    async def _ok():
        return {}

    assert inspect.iscoroutinefunction(_ok)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_public_data_http.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.internal.tools.public_data'`

- [ ] **Step 3: Create the package `__init__.py` placeholder**

Create `src/internal/tools/public_data/__init__.py`:

```python
"""Keyless public data-source tools (weather, markets, reference, geo).

Each theme module exposes ``build_*_tool()`` factories returning FunctionTools.
``public_data_tools()`` collects them; see ``knowledge_base.seed_tools``.
"""

from __future__ import annotations
```

- [ ] **Step 4: Write `_http.py`**

Create `src/internal/tools/public_data/_http.py`:

```python
"""Shared HTTP plumbing for the public data-source tools.

One place owns the timeout, the User-Agent, and the error shape, so each theme
module is only about its upstream's response format.

Absolute import of the aiohttp shim: this module sits one package deeper than
``tools.search``, and a four-dot relative import is needlessly hard to read.
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any, Callable

from src.context.retrieval.client import aiohttp

logger = logging.getLogger(__name__)

# Nominatim's usage policy requires an identifying User-Agent, and a generic
# browser string risks getting the whole project blocked. Individual callers
# may still override it (Yahoo rejects non-browser agents).
USER_AGENT = "AgenticSearch/1.0 (+https://github.com/linghypshen/Agentic-Search)"

DEFAULT_TIMEOUT_SECONDS = 10.0

# Upper bound on any single document body handed back to the model. Abstracts
# and article intros are otherwise long enough to crowd out the rollout budget.
MAX_CONTENT_CHARS = 1500


class PublicDataError(Exception):
    """An upstream call failed. ``guarded`` turns this into {"error": ...}."""


async def _fetch(
    method: str,
    url: str,
    *,
    params: dict | None = None,
    data: Any = None,
    headers: dict | None = None,
    timeout_seconds: float,
    as_json: bool,
) -> Any:
    merged = {"User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method, url, params=params, data=data, headers=merged
            ) as response:
                if response.status >= 400:
                    raise PublicDataError(f"{url} returned HTTP {response.status}")
                body = await response.text()
    except PublicDataError:
        raise
    except Exception as exc:
        logger.debug("public data request to %s failed", url, exc_info=True)
        raise PublicDataError(f"request to {url} failed: {exc}") from exc

    if not as_json:
        return body
    # Read text then parse, rather than response.json(): several of these hosts
    # return JSON under a non-JSON content type, which aiohttp rejects.
    try:
        return json.loads(body)
    except ValueError as exc:
        raise PublicDataError(f"{url} returned a non-JSON body") from exc


async def get_json(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    """GET *url* and parse the response as JSON. Raises PublicDataError."""
    return await _fetch(
        "GET",
        url,
        params=params,
        headers=headers,
        timeout_seconds=timeout_seconds,
        as_json=True,
    )


async def get_text(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """GET *url* and return the raw body. Raises PublicDataError."""
    return await _fetch(
        "GET",
        url,
        params=params,
        headers=headers,
        timeout_seconds=timeout_seconds,
        as_json=False,
    )


async def post_json(
    url: str,
    *,
    data: Any,
    headers: dict | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    """POST *data* to *url* and parse the response as JSON."""
    return await _fetch(
        "POST",
        url,
        data=data,
        headers=headers,
        timeout_seconds=timeout_seconds,
        as_json=True,
    )


def guarded(fn: Callable) -> Callable:
    """Adapt a tool coroutine to the tool return contract.

    The wrapped function returns a plain ``dict``/``list``; this serializes it
    and converts any failure into ``{"error": ...}`` so one dead upstream
    degrades a single tool rather than the whole turn.
    """

    @functools.wraps(fn)
    async def _wrapped(**kwargs: Any) -> str:
        try:
            return json.dumps(await fn(**kwargs))
        except PublicDataError as exc:
            return json.dumps({"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - a tool must never raise
            logger.debug("tool %s failed", getattr(fn, "__name__", "?"), exc_info=True)
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    return _wrapped
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/unit/test_public_data_http.py -v`
Expected: PASS (9 tests)

- [ ] **Step 6: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/tools/public_data/ tests/unit/test_public_data_http.py
git commit -m "feat(tools): add the public_data HTTP layer and error contract"
```

---

### Task 2: Knowledge tools — Wikipedia, ArXiv, Wayback

**Files:**
- Create: `src/internal/tools/public_data/knowledge.py`
- Test: `tests/unit/test_public_data_knowledge.py`

**Interfaces:**
- Consumes from Task 1: `get_json`, `get_text`, `guarded`, `PublicDataError`,
  `MAX_CONTENT_CHARS`.
- Produces, used by Task 5: `build_wikipedia_tool()`, `build_arxiv_tool()`,
  `build_wayback_tool()` — each `-> FunctionTool`, each `citeable=True`, named
  `search_wikipedia`, `search_arxiv`, `search_wayback`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_public_data_knowledge.py`:

```python
"""Unit tests for the Wikipedia / ArXiv / Wayback tools. No live network."""

from __future__ import annotations

import asyncio
import json

import pytest

from src.internal.tools.public_data import knowledge
from src.internal.tools.public_data._http import PublicDataError


def _run(tool, **arguments):
    response, _raw, _meta = asyncio.run(tool.execute("default", arguments))
    return json.loads(response)


ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <title>Dense Passage
      Retrieval</title>
    <summary>We study dense retrieval.</summary>
  </entry>
</feed>
"""

CDX_ROWS = [
    ["timestamp", "original", "statuscode", "mimetype"],
    ["20200101120000", "http://example.com/", "200", "text/html"],
]


def test_wikipedia_returns_citeable_shape(monkeypatch):
    calls = []

    async def _fake_get_json(url, *, params=None, **kwargs):
        calls.append(params)
        if params.get("list") == "search":
            return {"query": {"search": [{"pageid": 42, "title": "FAISS"}]}}
        return {"query": {"pages": {"42": {"title": "FAISS", "extract": "A library."}}}}

    monkeypatch.setattr(knowledge, "get_json", _fake_get_json)

    result = _run(knowledge.build_wikipedia_tool(), query="faiss")

    assert result == [
        {
            "title": "FAISS",
            "content": "A library.",
            "url": "https://en.wikipedia.org/?curid=42",
        }
    ]
    assert calls[0]["srsearch"] == "faiss"


def test_wikipedia_no_hits_returns_empty_list(monkeypatch):
    async def _fake_get_json(url, *, params=None, **kwargs):
        return {"query": {"search": []}}

    monkeypatch.setattr(knowledge, "get_json", _fake_get_json)

    assert _run(knowledge.build_wikipedia_tool(), query="zzzz") == []


def test_wikipedia_rejects_bogus_language(monkeypatch):
    async def _unreachable(*args, **kwargs):
        raise AssertionError("must not call out with an unvalidated language")

    monkeypatch.setattr(knowledge, "get_json", _unreachable)

    result = _run(knowledge.build_wikipedia_tool(), query="x", language="en.evil.com")

    assert "error" in result


def test_wikipedia_upstream_failure_returns_error(monkeypatch):
    async def _fail(*args, **kwargs):
        raise PublicDataError("wikipedia is down")

    monkeypatch.setattr(knowledge, "get_json", _fail)

    assert _run(knowledge.build_wikipedia_tool(), query="x") == {
        "error": "wikipedia is down"
    }


def test_arxiv_parses_atom_into_citeable_shape(monkeypatch):
    async def _fake_get_text(url, *, params=None, **kwargs):
        assert params["search_query"] == "all:retrieval"
        return ARXIV_FEED

    monkeypatch.setattr(knowledge, "get_text", _fake_get_text)

    result = _run(knowledge.build_arxiv_tool(), query="retrieval")

    assert result == [
        {
            "title": "Dense Passage Retrieval",
            "content": "We study dense retrieval.",
            "url": "http://arxiv.org/abs/1234.5678v1",
        }
    ]


def test_arxiv_malformed_feed_returns_error(monkeypatch):
    async def _fake_get_text(*args, **kwargs):
        return "not xml at all <"

    monkeypatch.setattr(knowledge, "get_text", _fake_get_text)

    assert "error" in _run(knowledge.build_arxiv_tool(), query="x")


def test_arxiv_upstream_failure_returns_error(monkeypatch):
    async def _fail(*args, **kwargs):
        raise PublicDataError("arxiv is down")

    monkeypatch.setattr(knowledge, "get_text", _fail)

    assert _run(knowledge.build_arxiv_tool(), query="x") == {"error": "arxiv is down"}


def test_wayback_returns_snapshot_rows(monkeypatch):
    async def _fake_get_json(url, *, params=None, **kwargs):
        assert params["url"] == "example.com"
        return CDX_ROWS

    monkeypatch.setattr(knowledge, "get_json", _fake_get_json)

    result = _run(knowledge.build_wayback_tool(), url="example.com")

    assert len(result) == 1
    assert result[0]["url"] == (
        "https://web.archive.org/web/20200101120000/http://example.com/"
    )
    assert "2020-01-01T12:00:00" in result[0]["title"]
    assert "200" in result[0]["content"]


def test_wayback_year_filter_is_sent(monkeypatch):
    captured = {}

    async def _fake_get_json(url, *, params=None, **kwargs):
        captured.update(params)
        return CDX_ROWS

    monkeypatch.setattr(knowledge, "get_json", _fake_get_json)
    _run(knowledge.build_wayback_tool(), url="example.com", year=2020)

    assert captured["from"] == "20200101"
    assert captured["to"] == "20201231"


def test_wayback_header_only_response_is_empty(monkeypatch):
    async def _fake_get_json(*args, **kwargs):
        return [["timestamp", "original", "statuscode", "mimetype"]]

    monkeypatch.setattr(knowledge, "get_json", _fake_get_json)

    assert _run(knowledge.build_wayback_tool(), url="nope.test") == []


def test_wayback_upstream_failure_returns_error(monkeypatch):
    async def _fail(*args, **kwargs):
        raise PublicDataError("archive is down")

    monkeypatch.setattr(knowledge, "get_json", _fail)

    assert _run(knowledge.build_wayback_tool(), url="x") == {"error": "archive is down"}


@pytest.mark.parametrize(
    "factory",
    [
        knowledge.build_wikipedia_tool,
        knowledge.build_arxiv_tool,
        knowledge.build_wayback_tool,
    ],
)
def test_knowledge_tools_are_citeable_and_read_only(factory):
    from src.internal.tools import ToolEffect

    tool = factory()
    assert tool.citeable is True
    assert tool.effect is ToolEffect.READ_ONLY
    assert tool.schema.description
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_public_data_knowledge.py -v`
Expected: FAIL — `ModuleNotFoundError: ... public_data.knowledge`

- [ ] **Step 3: Write `knowledge.py`**

Create `src/internal/tools/public_data/knowledge.py`:

```python
"""Reference-lookup tools: Wikipedia, ArXiv, and the Wayback Machine.

These three are the citeable ones: each returns a JSON array of
``{"title", "content", "url"}``, the same shape the corpus search returns, so
their results become source cards.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from xml.etree import ElementTree

from ..base import FunctionTool, ToolEffect
from ._http import MAX_CONTENT_CHARS, PublicDataError, get_json, get_text, guarded

logger = logging.getLogger(__name__)

WIKIPEDIA_API = "https://{language}.wikipedia.org/w/api.php"
ARXIV_API = "https://export.arxiv.org/api/query"
WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"

_ATOM = "{http://www.w3.org/2005/Atom}"
# The language code is interpolated into a hostname, so it is validated rather
# than escaped: anything outside this shape could redirect the request.
_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(-[a-z]{2,8})?$")
_ARXIV_SORTS = {"relevance", "lastUpdatedDate", "submittedDate"}


# ---------------------------------------------------------------- Wikipedia

_WIKIPEDIA_PARAMS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to look up."},
        "language": {
            "type": "string",
            "description": "Wikipedia language code, e.g. 'en' or 'de'.",
            "default": "en",
        },
        "limit": {
            "type": "integer",
            "description": "How many articles to return (1-10).",
            "default": 3,
        },
    },
    "required": ["query"],
}


async def _search_wikipedia(
    query: str, language: str = "en", limit: int = 3
) -> list[dict]:
    language = language.strip().lower()
    if not _LANGUAGE_RE.match(language):
        raise PublicDataError(f"unsupported Wikipedia language code {language!r}")
    api = WIKIPEDIA_API.format(language=language)
    limit = max(1, min(int(limit), 10))

    found = await get_json(
        api,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        },
    )
    hits = (found.get("query") or {}).get("search") or []
    if not hits:
        return []

    page_ids = [str(hit["pageid"]) for hit in hits if hit.get("pageid")]
    extracts = await get_json(
        api,
        params={
            "action": "query",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "pageids": "|".join(page_ids),
            "format": "json",
        },
    )
    pages = ((extracts.get("query") or {}).get("pages")) or {}
    results = []
    for page_id in page_ids:
        page = pages.get(page_id) or {}
        results.append(
            {
                "title": page.get("title", ""),
                "content": (page.get("extract") or "")[:MAX_CONTENT_CHARS],
                # ?curid= is the stable per-page URL and needs no title escaping.
                "url": f"https://{language}.wikipedia.org/?curid={page_id}",
            }
        )
    return results


def build_wikipedia_tool() -> FunctionTool:
    return FunctionTool(
        fn=guarded(_search_wikipedia),
        name="search_wikipedia",
        description=(
            "Look up encyclopedia articles on Wikipedia. Use for background on "
            "people, places, organizations, events, and general concepts."
        ),
        parameters=_WIKIPEDIA_PARAMS,
        effect=ToolEffect.READ_ONLY,
        citeable=True,
    )


# -------------------------------------------------------------------- ArXiv

_ARXIV_PARAMS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Topic, title, or author."},
        "max_results": {
            "type": "integer",
            "description": "How many papers to return (1-25).",
            "default": 5,
        },
        "sort_by": {
            "type": "string",
            "description": "One of relevance, lastUpdatedDate, submittedDate.",
            "default": "relevance",
        },
    },
    "required": ["query"],
}


async def _search_arxiv(
    query: str, max_results: int = 5, sort_by: str = "relevance"
) -> list[dict]:
    body = await get_text(
        ARXIV_API,
        params={
            "search_query": f"all:{query}",
            "max_results": max(1, min(int(max_results), 25)),
            "sortBy": sort_by if sort_by in _ARXIV_SORTS else "relevance",
            "sortOrder": "descending",
        },
    )
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as exc:
        raise PublicDataError("arxiv returned a malformed Atom feed") from exc

    results = []
    for entry in root.findall(f"{_ATOM}entry"):
        title = entry.findtext(f"{_ATOM}title") or ""
        summary = entry.findtext(f"{_ATOM}summary") or ""
        results.append(
            {
                # Atom wraps titles across lines; collapse the whitespace.
                "title": " ".join(title.split()),
                "content": " ".join(summary.split())[:MAX_CONTENT_CHARS],
                "url": (entry.findtext(f"{_ATOM}id") or "").strip(),
            }
        )
    return results


def build_arxiv_tool() -> FunctionTool:
    return FunctionTool(
        fn=guarded(_search_arxiv),
        name="search_arxiv",
        description=(
            "Search ArXiv for academic preprints and return their abstracts. "
            "Use for research papers, methods, and scientific results."
        ),
        parameters=_ARXIV_PARAMS,
        effect=ToolEffect.READ_ONLY,
        citeable=True,
    )


# ----------------------------------------------------------------- Wayback

_WAYBACK_PARAMS = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "The URL to look up."},
        "year": {
            "type": "integer",
            "description": "Restrict snapshots to this calendar year.",
        },
        "limit": {
            "type": "integer",
            "description": "How many snapshots to return (1-50).",
            "default": 10,
        },
    },
    "required": ["url"],
}


async def _search_wayback(
    url: str, year: int | None = None, limit: int = 10
) -> list[dict]:
    params = {
        "url": url,
        "output": "json",
        "limit": max(1, min(int(limit), 50)),
        "fl": "timestamp,original,statuscode,mimetype",
    }
    if year:
        params["from"] = f"{int(year)}0101"
        params["to"] = f"{int(year)}1231"

    rows = await get_json(WAYBACK_CDX, params=params)
    # The CDX API answers with a header row followed by data rows; a lone
    # header row means "archived nothing".
    if not isinstance(rows, list) or len(rows) < 2:
        return []

    headers, *data = rows
    results = []
    for row in data:
        record = dict(zip(headers, row))
        stamp = record.get("timestamp", "")
        try:
            when = datetime.strptime(stamp, "%Y%m%d%H%M%S").isoformat()
        except ValueError:
            when = stamp
        original = record.get("original", url)
        results.append(
            {
                "title": f"Snapshot of {original} at {when}",
                "content": (
                    f"HTTP {record.get('statuscode', '?')}, "
                    f"{record.get('mimetype', 'unknown')}"
                ),
                "url": f"https://web.archive.org/web/{stamp}/{original}",
            }
        )
    return results


def build_wayback_tool() -> FunctionTool:
    return FunctionTool(
        fn=guarded(_search_wayback),
        name="search_wayback",
        description=(
            "Find archived snapshots of a web page in the Internet Archive's "
            "Wayback Machine. Use to see what a URL looked like in the past."
        ),
        parameters=_WAYBACK_PARAMS,
        effect=ToolEffect.READ_ONLY,
        citeable=True,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_public_data_knowledge.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/tools/public_data/knowledge.py tests/unit/test_public_data_knowledge.py
git commit -m "feat(tools): add citeable Wikipedia, ArXiv, and Wayback tools"
```

---

### Task 3: Market tools — stocks, crypto, currency

**Files:**
- Create: `src/internal/tools/public_data/market.py`
- Test: `tests/unit/test_public_data_market.py`

**Interfaces:**
- Consumes from Task 1: `get_json`, `guarded`, `PublicDataError`.
- Produces, used by Task 5: `build_stock_quote_tool()`,
  `build_crypto_price_tool()`, `build_currency_tool()` — each
  `-> FunctionTool`, all `citeable=False`, named `get_stock_quote`,
  `get_crypto_price`, `convert_currency`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_public_data_market.py`:

```python
"""Unit tests for the stock / crypto / currency tools. No live network."""

from __future__ import annotations

import asyncio
import json

from src.internal.tools.public_data import market
from src.internal.tools.public_data._http import PublicDataError


def _run(tool, **arguments):
    response, _raw, _meta = asyncio.run(tool.execute("default", arguments))
    return json.loads(response)


def test_stock_quote_maps_yahoo_meta(monkeypatch):
    captured = {}

    async def _fake_get_json(url, *, params=None, headers=None, **kwargs):
        captured["url"] = url
        captured["headers"] = headers
        return {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "currency": "USD",
                            "regularMarketPrice": 210.5,
                            "previousClose": 208.0,
                            "regularMarketDayHigh": 211.0,
                            "regularMarketDayLow": 207.5,
                            "regularMarketVolume": 1234,
                            "exchangeName": "NMS",
                        }
                    }
                ]
            }
        }

    monkeypatch.setattr(market, "get_json", _fake_get_json)

    result = _run(market.build_stock_quote_tool(), symbol="aapl")

    assert result["symbol"] == "AAPL"
    assert result["current_price"] == 210.5
    assert result["exchange"] == "NMS"
    assert captured["url"].endswith("/aapl")
    # Yahoo 4xxs a request without a browser-like agent.
    assert "Mozilla" in captured["headers"]["User-Agent"]


def test_stock_quote_empty_result_is_an_error(monkeypatch):
    async def _fake_get_json(*args, **kwargs):
        return {"chart": {"result": []}}

    monkeypatch.setattr(market, "get_json", _fake_get_json)

    assert "error" in _run(market.build_stock_quote_tool(), symbol="ZZZZ")


def test_stock_quote_rejects_path_traversal(monkeypatch):
    async def _unreachable(*args, **kwargs):
        raise AssertionError("must not call out with an unvalidated symbol")

    monkeypatch.setattr(market, "get_json", _unreachable)

    assert "error" in _run(market.build_stock_quote_tool(), symbol="../../evil")


def test_crypto_price_maps_symbol_to_coingecko_id(monkeypatch):
    captured = {}

    async def _fake_get_json(url, *, params=None, **kwargs):
        captured.update(params)
        return {
            "bitcoin": {
                "usd": 64000.0,
                "usd_market_cap": 1.2e12,
                "usd_24h_vol": 3.0e10,
                "usd_24h_change": -1.5,
                "last_updated_at": 1700000000,
            }
        }

    monkeypatch.setattr(market, "get_json", _fake_get_json)

    result = _run(market.build_crypto_price_tool(), symbol="btc")

    assert captured["ids"] == "bitcoin"
    assert result["coin_id"] == "bitcoin"
    assert result["price"] == 64000.0
    assert result["change_24h_percent"] == -1.5
    assert result["last_updated"].startswith("20")


def test_crypto_price_unknown_coin_is_an_error(monkeypatch):
    async def _fake_get_json(*args, **kwargs):
        return {}

    monkeypatch.setattr(market, "get_json", _fake_get_json)

    assert "error" in _run(market.build_crypto_price_tool(), symbol="notacoin")


def test_convert_currency_computes_amount(monkeypatch):
    async def _fake_get_json(url, **kwargs):
        assert url.endswith("/USD")
        return {"rates": {"EUR": 0.9}, "date": "2026-08-06"}

    monkeypatch.setattr(market, "get_json", _fake_get_json)

    result = _run(
        market.build_currency_tool(),
        amount=10.0,
        from_currency="usd",
        to_currency="eur",
    )

    assert result["converted_amount"] == 9.0
    assert result["rate"] == 0.9
    assert result["from_currency"] == "USD"


def test_convert_currency_unknown_target_is_an_error(monkeypatch):
    async def _fake_get_json(*args, **kwargs):
        return {"rates": {"EUR": 0.9}}

    monkeypatch.setattr(market, "get_json", _fake_get_json)

    result = _run(
        market.build_currency_tool(),
        amount=1.0,
        from_currency="USD",
        to_currency="XYZ",
    )
    assert "error" in result


def test_convert_currency_rejects_non_iso_code(monkeypatch):
    async def _unreachable(*args, **kwargs):
        raise AssertionError("must not call out with an unvalidated code")

    monkeypatch.setattr(market, "get_json", _unreachable)

    result = _run(
        market.build_currency_tool(),
        amount=1.0,
        from_currency="../secrets",
        to_currency="EUR",
    )
    assert "error" in result


def test_currency_upstream_failure_returns_error(monkeypatch):
    async def _fail(*args, **kwargs):
        raise PublicDataError("rates are down")

    monkeypatch.setattr(market, "get_json", _fail)

    result = _run(
        market.build_currency_tool(),
        amount=1.0,
        from_currency="USD",
        to_currency="EUR",
    )
    assert result == {"error": "rates are down"}


def test_market_tools_are_not_citeable():
    for factory in (
        market.build_stock_quote_tool,
        market.build_crypto_price_tool,
        market.build_currency_tool,
    ):
        assert factory().citeable is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_public_data_market.py -v`
Expected: FAIL — `ModuleNotFoundError: ... public_data.market`

- [ ] **Step 3: Write `market.py`**

Create `src/internal/tools/public_data/market.py`:

```python
"""Market-data tools: stock quotes, crypto prices, currency conversion.

All three answer with a flat JSON object of facts rather than documents, so
none of them is citeable.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from ..base import FunctionTool, ToolEffect
from ._http import PublicDataError, get_json, guarded

logger = logging.getLogger(__name__)

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
COINGECKO_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"
EXCHANGE_RATE_URL = "https://api.exchangerate-api.com/v4/latest/{base}"

# Yahoo 4xxs anything that does not look like a browser.
_YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Both of these are interpolated into a URL path, so they are validated rather
# than escaped — a stray "../" would otherwise retarget the request.
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9.^=-]{1,15}$")
_CURRENCY_RE = re.compile(r"^[A-Za-z]{3}$")

# CoinGecko keys on slugs, not tickers; models say "btc".
_COIN_IDS = {
    "btc": "bitcoin",
    "eth": "ethereum",
    "usdt": "tether",
    "bnb": "binancecoin",
    "sol": "solana",
    "xrp": "ripple",
    "usdc": "usd-coin",
    "ada": "cardano",
    "doge": "dogecoin",
    "trx": "tron",
    "dot": "polkadot",
    "matic": "matic-network",
    "dai": "dai",
    "shib": "shiba-inu",
    "avax": "avalanche-2",
}


# ------------------------------------------------------------------- stocks

_STOCK_PARAMS = {
    "type": "object",
    "properties": {
        "symbol": {
            "type": "string",
            "description": "Ticker symbol, e.g. AAPL or TSLA.",
        },
    },
    "required": ["symbol"],
}


async def _get_stock_quote(symbol: str) -> dict:
    symbol = symbol.strip()
    if not _SYMBOL_RE.match(symbol):
        raise PublicDataError(f"invalid ticker symbol {symbol!r}")

    payload = await get_json(
        YAHOO_CHART_URL.format(symbol=symbol),
        params={"interval": "1d", "range": "1d"},
        headers=_YAHOO_HEADERS,
    )
    results = (payload.get("chart") or {}).get("result") or []
    if not results:
        raise PublicDataError(f"no quote data for symbol {symbol!r}")

    meta = results[0].get("meta") or {}
    return {
        "symbol": symbol.upper(),
        "currency": meta.get("currency", "USD"),
        "current_price": meta.get("regularMarketPrice"),
        "previous_close": meta.get("previousClose") or meta.get("chartPreviousClose"),
        "day_high": meta.get("regularMarketDayHigh"),
        "day_low": meta.get("regularMarketDayLow"),
        "volume": meta.get("regularMarketVolume"),
        "exchange": meta.get("exchangeName"),
    }


def build_stock_quote_tool() -> FunctionTool:
    return FunctionTool(
        fn=guarded(_get_stock_quote),
        name="get_stock_quote",
        description=(
            "Get the latest price, day range, and volume for a listed stock or "
            "ETF by ticker symbol."
        ),
        parameters=_STOCK_PARAMS,
        effect=ToolEffect.READ_ONLY,
    )


# ------------------------------------------------------------------- crypto

_CRYPTO_PARAMS = {
    "type": "object",
    "properties": {
        "symbol": {
            "type": "string",
            "description": "Coin ticker or CoinGecko id, e.g. btc or bitcoin.",
        },
        "vs_currency": {
            "type": "string",
            "description": "Currency to price in, e.g. usd or eur.",
            "default": "usd",
        },
    },
    "required": ["symbol"],
}


async def _get_crypto_price(symbol: str, vs_currency: str = "usd") -> dict:
    key = symbol.strip().lower()
    coin_id = _COIN_IDS.get(key, key)
    vs = vs_currency.strip().lower()

    payload = await get_json(
        COINGECKO_PRICE_URL,
        params={
            "ids": coin_id,
            "vs_currencies": vs,
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_last_updated_at": "true",
        },
    )
    data = (payload or {}).get(coin_id)
    if not data:
        raise PublicDataError(f"unknown cryptocurrency {symbol!r}")

    updated = data.get("last_updated_at")
    return {
        "symbol": symbol.upper(),
        "coin_id": coin_id,
        "currency": vs.upper(),
        "price": data.get(vs),
        "market_cap": data.get(f"{vs}_market_cap"),
        "volume_24h": data.get(f"{vs}_24h_vol"),
        "change_24h_percent": data.get(f"{vs}_24h_change"),
        "last_updated": (
            datetime.fromtimestamp(updated, tz=timezone.utc).isoformat()
            if updated
            else None
        ),
    }


def build_crypto_price_tool() -> FunctionTool:
    return FunctionTool(
        fn=guarded(_get_crypto_price),
        name="get_crypto_price",
        description=(
            "Get the current price, market cap, and 24-hour change for a "
            "cryptocurrency such as bitcoin or ethereum."
        ),
        parameters=_CRYPTO_PARAMS,
        effect=ToolEffect.READ_ONLY,
    )


# ----------------------------------------------------------------- currency

_CURRENCY_PARAMS = {
    "type": "object",
    "properties": {
        "amount": {"type": "number", "description": "How much to convert."},
        "from_currency": {
            "type": "string",
            "description": "Three-letter source currency code, e.g. USD.",
        },
        "to_currency": {
            "type": "string",
            "description": "Three-letter target currency code, e.g. EUR.",
        },
    },
    "required": ["amount", "from_currency", "to_currency"],
}


def _currency_code(value: str, label: str) -> str:
    code = value.strip()
    if not _CURRENCY_RE.match(code):
        raise PublicDataError(f"{label} must be a 3-letter currency code, got {value!r}")
    return code.upper()


async def _convert_currency(
    amount: float, from_currency: str, to_currency: str
) -> dict:
    base = _currency_code(from_currency, "from_currency")
    target = _currency_code(to_currency, "to_currency")

    payload = await get_json(EXCHANGE_RATE_URL.format(base=base))
    rates = payload.get("rates") or {}
    if target not in rates:
        raise PublicDataError(f"no exchange rate from {base} to {target}")

    rate = rates[target]
    return {
        "amount": float(amount),
        "from_currency": base,
        "to_currency": target,
        "rate": rate,
        "converted_amount": round(float(amount) * rate, 6),
        "date": payload.get("date"),
    }


def build_currency_tool() -> FunctionTool:
    return FunctionTool(
        fn=guarded(_convert_currency),
        name="convert_currency",
        description=(
            "Convert an amount of money from one currency to another at "
            "today's exchange rate."
        ),
        parameters=_CURRENCY_PARAMS,
        effect=ToolEffect.READ_ONLY,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_public_data_market.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/tools/public_data/market.py tests/unit/test_public_data_market.py
git commit -m "feat(tools): add stock quote, crypto price, and currency tools"
```

---

### Task 4: Geo tools — weather, geocoding, nearby places

**Files:**
- Create: `src/internal/tools/public_data/geo.py`
- Test: `tests/unit/test_public_data_geo.py`

**Interfaces:**
- Consumes from Task 1: `get_json`, `post_json`, `guarded`, `PublicDataError`.
- Produces, used by Task 5: `build_weather_tool()`, `build_location_tool()`,
  `build_nearby_places_tool()` — each `-> FunctionTool`, all `citeable=False`,
  named `get_weather`, `search_location`, `search_nearby_places`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_public_data_geo.py`:

```python
"""Unit tests for the weather / geocoding / POI tools. No live network."""

from __future__ import annotations

import asyncio
import json

from src.internal.tools.public_data import geo
from src.internal.tools.public_data._http import PublicDataError


def _run(tool, **arguments):
    response, _raw, _meta = asyncio.run(tool.execute("default", arguments))
    return json.loads(response)


_FORECAST = {
    "current": {
        "time": "2026-08-06T12:00",
        "temperature_2m": 14.2,
        "apparent_temperature": 13.0,
        "relative_humidity_2m": 71,
        "precipitation": 0.0,
        "weather_code": 3,
        "wind_speed_10m": 11.5,
        "wind_direction_10m": 240,
    }
}


def test_weather_geocodes_then_fetches(monkeypatch):
    urls = []

    async def _fake_get_json(url, *, params=None, **kwargs):
        urls.append(url)
        if "geocoding" in url:
            return {
                "results": [
                    {"latitude": 51.5, "longitude": -0.13, "name": "London",
                     "country": "United Kingdom"}
                ]
            }
        return _FORECAST

    monkeypatch.setattr(geo, "get_json", _fake_get_json)

    result = _run(geo.build_weather_tool(), location="london")

    assert len(urls) == 2
    assert result["location"] == "London"
    assert result["country"] == "United Kingdom"
    assert result["temperature"] == 14.2
    assert result["description"] == "Overcast"


def test_weather_skips_geocoding_when_coordinates_given(monkeypatch):
    urls = []

    async def _fake_get_json(url, *, params=None, **kwargs):
        urls.append(url)
        return _FORECAST

    monkeypatch.setattr(geo, "get_json", _fake_get_json)

    result = _run(
        geo.build_weather_tool(), location="Somewhere", latitude=1.0, longitude=2.0
    )

    assert len(urls) == 1
    assert result["latitude"] == 1.0


def test_weather_unknown_location_is_an_error(monkeypatch):
    async def _fake_get_json(url, *, params=None, **kwargs):
        return {"results": []}

    monkeypatch.setattr(geo, "get_json", _fake_get_json)

    assert "error" in _run(geo.build_weather_tool(), location="zzzzz")


def test_weather_unknown_code_falls_back(monkeypatch):
    async def _fake_get_json(url, *, params=None, **kwargs):
        if "geocoding" in url:
            return {"results": [{"latitude": 0.0, "longitude": 0.0, "name": "X"}]}
        return {"current": dict(_FORECAST["current"], weather_code=999)}

    monkeypatch.setattr(geo, "get_json", _fake_get_json)

    assert _run(geo.build_weather_tool(), location="X")["description"] == "Unknown"


def test_search_location_flattens_nominatim(monkeypatch):
    captured = {}

    async def _fake_get_json(url, *, params=None, headers=None, **kwargs):
        captured["params"] = params
        return [
            {
                "display_name": "Eiffel Tower, Paris, France",
                "lat": "48.858",
                "lon": "2.294",
                "type": "attraction",
                "class": "tourism",
                "address": {"country": "France", "city": "Paris"},
            }
        ]

    monkeypatch.setattr(geo, "get_json", _fake_get_json)

    result = _run(geo.build_location_tool(), query="eiffel tower", country_code="FR")

    assert result["count"] == 1
    place = result["locations"][0]
    assert place["latitude"] == 48.858
    assert place["city"] == "Paris"
    assert captured["params"]["countrycodes"] == "fr"


def test_search_location_no_results(monkeypatch):
    async def _fake_get_json(*args, **kwargs):
        return []

    monkeypatch.setattr(geo, "get_json", _fake_get_json)

    assert _run(geo.build_location_tool(), query="zzz")["count"] == 0


def test_nearby_places_parses_overpass_elements(monkeypatch):
    captured = {}

    async def _fake_post_json(url, *, data=None, **kwargs):
        captured["data"] = data
        return {
            "elements": [
                {
                    "id": 1,
                    "lat": 48.86,
                    "lon": 2.29,
                    "tags": {
                        "name": "Cafe Central",
                        "amenity": "cafe",
                        "addr:street": "Rue X",
                        "opening_hours": "08:00-18:00",
                    },
                }
            ]
        }

    monkeypatch.setattr(geo, "post_json", _fake_post_json)

    result = _run(
        geo.build_nearby_places_tool(),
        query="cafe",
        latitude=48.86,
        longitude=2.29,
    )

    assert result["count"] == 1
    assert result["places"][0]["name"] == "Cafe Central"
    assert result["places"][0]["type"] == "cafe"
    # Anchored so "cafe" does not also match "cafeteria_supplier".
    assert '~"^cafe$"' in captured["data"]["data"]


def test_nearby_places_rejects_overpass_injection(monkeypatch):
    async def _unreachable(*args, **kwargs):
        raise AssertionError("must not build a query from unvalidated input")

    monkeypatch.setattr(geo, "post_json", _unreachable)

    result = _run(
        geo.build_nearby_places_tool(),
        query='cafe"](around:1,0,0);out;//',
        latitude=0.0,
        longitude=0.0,
    )
    assert "error" in result


def test_nearby_places_upstream_failure_returns_error(monkeypatch):
    async def _fail(*args, **kwargs):
        raise PublicDataError("overpass is busy")

    monkeypatch.setattr(geo, "post_json", _fail)

    result = _run(
        geo.build_nearby_places_tool(), query="cafe", latitude=0.0, longitude=0.0
    )
    assert result == {"error": "overpass is busy"}


def test_geo_tools_are_not_citeable():
    for factory in (
        geo.build_weather_tool,
        geo.build_location_tool,
        geo.build_nearby_places_tool,
    ):
        assert factory().citeable is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_public_data_geo.py -v`
Expected: FAIL — `ModuleNotFoundError: ... public_data.geo`

- [ ] **Step 3: Write `geo.py`**

Create `src/internal/tools/public_data/geo.py`:

```python
"""Location tools: current weather, geocoding, and nearby points of interest.

All three answer with a flat JSON object of facts, so none is citeable.
"""

from __future__ import annotations

import logging
import re

from ..base import FunctionTool, ToolEffect
from ._http import PublicDataError, get_json, guarded, post_json

logger = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Overpass queries are slow; give them their own budget.
OVERPASS_TIMEOUT_SECONDS = 30.0

# The place type is interpolated into an Overpass QL regex. Validate rather
# than escape: a stray `"]` would otherwise close the tag filter and let the
# caller append arbitrary statements to the query.
_PLACE_TYPE_RE = re.compile(r"^[A-Za-z0-9 _-]{1,40}$")

# WMO weather interpretation codes.
_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


# ------------------------------------------------------------------ weather

_WEATHER_PARAMS = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "City or place name, e.g. 'Berlin'.",
        },
        "latitude": {
            "type": "number",
            "description": "Latitude; skips the place-name lookup when given.",
        },
        "longitude": {
            "type": "number",
            "description": "Longitude; skips the place-name lookup when given.",
        },
    },
    "required": ["location"],
}


async def _get_weather(
    location: str,
    latitude: float | None = None,
    longitude: float | None = None,
) -> dict:
    country = ""
    if latitude is None or longitude is None:
        found = await get_json(
            GEOCODE_URL,
            params={
                "name": location,
                "count": 1,
                "language": "en",
                "format": "json",
            },
        )
        results = found.get("results") or []
        if not results:
            raise PublicDataError(f"could not find a place called {location!r}")
        first = results[0]
        latitude = first["latitude"]
        longitude = first["longitude"]
        location = first.get("name", location)
        country = first.get("country", "")

    payload = await get_json(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,weather_code,wind_speed_10m,wind_direction_10m"
            ),
            "timezone": "auto",
        },
    )
    current = payload.get("current") or {}
    return {
        "location": location,
        "country": country,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "temperature": current.get("temperature_2m"),
        "feels_like": current.get("apparent_temperature"),
        "humidity": current.get("relative_humidity_2m"),
        "precipitation": current.get("precipitation"),
        "description": _WEATHER_CODES.get(current.get("weather_code"), "Unknown"),
        "wind_speed": current.get("wind_speed_10m"),
        "wind_direction": current.get("wind_direction_10m"),
        "units": "metric",
        "observed_at": current.get("time"),
    }


def build_weather_tool() -> FunctionTool:
    return FunctionTool(
        fn=guarded(_get_weather),
        name="get_weather",
        description=(
            "Get the current weather — temperature, conditions, wind, humidity "
            "— for a city or place."
        ),
        parameters=_WEATHER_PARAMS,
        effect=ToolEffect.READ_ONLY,
    )


# ----------------------------------------------------------------- geocoding

_LOCATION_PARAMS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Place, address, or landmark to find.",
        },
        "limit": {
            "type": "integer",
            "description": "How many matches to return (1-20).",
            "default": 5,
        },
        "country_code": {
            "type": "string",
            "description": "Two-letter country filter, e.g. 'fr'.",
        },
    },
    "required": ["query"],
}


async def _search_location(
    query: str, limit: int = 5, country_code: str | None = None
) -> dict:
    params = {
        "q": query,
        "format": "json",
        "limit": max(1, min(int(limit), 20)),
        "addressdetails": 1,
    }
    if country_code:
        params["countrycodes"] = country_code.strip().lower()

    rows = await get_json(NOMINATIM_URL, params=params)
    locations = []
    for item in rows or []:
        address = item.get("address") or {}
        locations.append(
            {
                "display_name": item.get("display_name"),
                "latitude": float(item["lat"]),
                "longitude": float(item["lon"]),
                "type": item.get("type"),
                "category": item.get("class"),
                "country": address.get("country"),
                "city": (
                    address.get("city")
                    or address.get("town")
                    or address.get("village")
                ),
                "postcode": address.get("postcode"),
            }
        )
    return {"query": query, "locations": locations, "count": len(locations)}


def build_location_tool() -> FunctionTool:
    return FunctionTool(
        fn=guarded(_search_location),
        name="search_location",
        description=(
            "Find the coordinates and address of a place, landmark, or address "
            "by name. Use this before search_nearby_places."
        ),
        parameters=_LOCATION_PARAMS,
        effect=ToolEffect.READ_ONLY,
    )


# -------------------------------------------------------------------- places

_PLACES_PARAMS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Place type, e.g. 'cafe', 'pharmacy', 'hotel'.",
        },
        "latitude": {"type": "number", "description": "Centre latitude."},
        "longitude": {"type": "number", "description": "Centre longitude."},
        "radius_meters": {
            "type": "integer",
            "description": "Search radius in metres (max 10000).",
            "default": 1000,
        },
        "limit": {
            "type": "integer",
            "description": "How many places to return (1-50).",
            "default": 10,
        },
    },
    "required": ["query", "latitude", "longitude"],
}


async def _search_nearby_places(
    query: str,
    latitude: float,
    longitude: float,
    radius_meters: int = 1000,
    limit: int = 10,
) -> dict:
    place_type = query.strip()
    if not _PLACE_TYPE_RE.match(place_type):
        raise PublicDataError(
            "place type must be 1-40 characters of letters, digits, spaces, "
            "hyphens, or underscores (for example 'cafe' or 'fire station')"
        )
    lat = float(latitude)
    lon = float(longitude)
    radius = max(1, min(int(radius_meters), 10000))
    limit = max(1, min(int(limit), 50))

    # Anchored so "cafe" matches amenity=cafe and not every value containing
    # it. The name-substring clause the reference used is dropped: it makes
    # Overpass scan far more elements and routinely times out.
    around = f"(around:{radius},{lat},{lon})"
    overpass_ql = (
        "[out:json][timeout:25];\n"
        "(\n"
        f'  node["amenity"~"^{place_type}$",i]{around};\n'
        f'  node["shop"~"^{place_type}$",i]{around};\n'
        f'  node["tourism"~"^{place_type}$",i]{around};\n'
        ");\n"
        f"out body {limit};"
    )

    payload = await post_json(
        OVERPASS_URL,
        data={"data": overpass_ql},
        timeout_seconds=OVERPASS_TIMEOUT_SECONDS,
    )
    places = []
    for element in (payload.get("elements") or [])[:limit]:
        tags = element.get("tags") or {}
        places.append(
            {
                "name": tags.get("name", "Unnamed"),
                "type": tags.get("amenity") or tags.get("shop") or tags.get("tourism"),
                "latitude": element.get("lat"),
                "longitude": element.get("lon"),
                "street": tags.get("addr:street"),
                "city": tags.get("addr:city"),
                "phone": tags.get("phone"),
                "website": tags.get("website"),
                "opening_hours": tags.get("opening_hours"),
            }
        )
    return {
        "query": place_type,
        "centre": {"latitude": lat, "longitude": lon},
        "radius_meters": radius,
        "places": places,
        "count": len(places),
    }


def build_nearby_places_tool() -> FunctionTool:
    return FunctionTool(
        fn=guarded(_search_nearby_places),
        name="search_nearby_places",
        description=(
            "List points of interest of a given type — cafes, pharmacies, "
            "hotels — near a latitude and longitude."
        ),
        parameters=_PLACES_PARAMS,
        effect=ToolEffect.READ_ONLY,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_public_data_geo.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/tools/public_data/geo.py tests/unit/test_public_data_geo.py
git commit -m "feat(tools): add weather, geocoding, and nearby-places tools"
```

---

### Task 5: Collect the nine and seed them

**Files:**
- Modify: `src/internal/tools/public_data/__init__.py` (replace the Task 1 placeholder body)
- Modify: `src/internal/tools/knowledge_base.py:38-51`
- Test: `tests/unit/test_public_data_seeding.py`

**Interfaces:**
- Consumes from Tasks 2–4: the nine `build_*_tool()` factories.
- Produces, used by Task 6: `public_data_tools() -> list[Tool]` returning the
  nine in a fixed order; `tool_knowledge_base()` includes them.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_public_data_seeding.py`:

```python
"""The nine public data tools are collected and seeded as agent-callable."""

from __future__ import annotations

from src.internal.tools.public_data import public_data_tools
from src.internal.tools.knowledge_base import (
    NOT_AGENT_CALLABLE,
    seed_tools,
    tool_knowledge_base,
)
from src.internal.tools.registry import ToolRegistry
from src.internal.tools.validation import validate_arguments

EXPECTED = {
    "search_wikipedia",
    "search_arxiv",
    "search_wayback",
    "get_weather",
    "get_stock_quote",
    "get_crypto_price",
    "convert_currency",
    "search_location",
    "search_nearby_places",
}


def test_public_data_tools_returns_the_nine():
    names = [t.name for t in public_data_tools()]
    assert len(names) == 9
    assert set(names) == EXPECTED
    assert len(set(names)) == len(names)


def test_every_tool_has_a_usable_schema():
    for tool in public_data_tools():
        params = tool.schema.parameters
        assert params["type"] == "object"
        assert params["properties"]
        assert tool.schema.description
        # Required names must exist in properties, or validation can never pass.
        for name in params.get("required", []):
            assert name in params["properties"]


def test_schema_required_fields_are_enforced():
    tool = next(t for t in public_data_tools() if t.name == "get_weather")
    assert validate_arguments(tool.schema.parameters, {}) != []
    assert validate_arguments(tool.schema.parameters, {"location": "Berlin"}) == []


def test_knowledge_base_includes_the_public_data_tools():
    names = {t.name for t in tool_knowledge_base()}
    assert EXPECTED <= names


def test_seeded_public_data_tools_are_agent_callable():
    registry = ToolRegistry()
    seed_tools(registry, tools=tool_knowledge_base())
    callable_names = {t.name for t in registry.agent_tools()}
    assert EXPECTED <= callable_names
    assert EXPECTED.isdisjoint(NOT_AGENT_CALLABLE)


def test_only_the_knowledge_tools_are_citeable():
    citeable = {t.name for t in public_data_tools() if t.citeable}
    assert citeable == {"search_wikipedia", "search_arxiv", "search_wayback"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_public_data_seeding.py -v`
Expected: FAIL — `ImportError: cannot import name 'public_data_tools'`

- [ ] **Step 3: Fill in `public_data/__init__.py`**

Replace the contents of `src/internal/tools/public_data/__init__.py`:

```python
"""Keyless public data-source tools (reference, markets, geo).

Every tool here reaches a free public API that needs no credentials, so they
work on a clean checkout with no configuration. ``public_data_tools()`` is the
set ``knowledge_base.tool_knowledge_base()`` seeds.

Return contract: the citeable tools (Wikipedia, ArXiv, Wayback) answer with a
JSON array of ``{"title", "content", "url"}`` — the same shape the corpus
search returns, which is what lets their results become source cards. Every
other tool answers with a JSON object of facts. Failures are
``{"error": ...}``; no tool raises.
"""

from __future__ import annotations

from ..base import Tool
from .geo import (
    build_location_tool,
    build_nearby_places_tool,
    build_weather_tool,
)
from .knowledge import (
    build_arxiv_tool,
    build_wayback_tool,
    build_wikipedia_tool,
)
from .market import (
    build_crypto_price_tool,
    build_currency_tool,
    build_stock_quote_tool,
)


def public_data_tools() -> list[Tool]:
    """Build the nine public data-source tools, in catalog order."""
    return [
        build_wikipedia_tool(),
        build_arxiv_tool(),
        build_wayback_tool(),
        build_weather_tool(),
        build_stock_quote_tool(),
        build_crypto_price_tool(),
        build_currency_tool(),
        build_location_tool(),
        build_nearby_places_tool(),
    ]


__all__ = ["public_data_tools"]
```

- [ ] **Step 4: Seed them from `knowledge_base.py`**

In `src/internal/tools/knowledge_base.py`, add the import next to the existing
relative imports:

```python
from .public_data import public_data_tools
```

Then extend the list built in `tool_knowledge_base()`. The existing code is:

```python
    tools: list[Tool] = [
        MultiQueryWebSearchTool(
            search_fn=make_web_cascade_search(
                browser_search_url=os.getenv("AGENTIC_SEARCH_BROWSER_SEARCH_URL")
            ),
            page_size=top_k,
        ),
        build_search_routing_tool(search_url=search_url, top_k=top_k),
    ]
    if llm is not None:
```

Change it to:

```python
    tools: list[Tool] = [
        MultiQueryWebSearchTool(
            search_fn=make_web_cascade_search(
                browser_search_url=os.getenv("AGENTIC_SEARCH_BROWSER_SEARCH_URL")
            ),
            page_size=top_k,
        ),
        build_search_routing_tool(search_url=search_url, top_k=top_k),
        # Keyless public data sources. They need no configuration, so they are
        # on by default: without them the tool agent has one usable tool and
        # nothing to choose between.
        *public_data_tools(),
    ]
    if llm is not None:
```

Do **not** add any of the nine to `NOT_AGENT_CALLABLE`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/unit/test_public_data_seeding.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Check nothing that counted the old seed set broke**

Run: `pytest tests/unit/test_built_in_tools.py tests/unit/test_tool_registry.py tests/unit/test_agent_callable_tools.py tests/unit/test_tool_categories.py tests/unit/test_tools_package_layout.py -v`
Expected: PASS. If a test asserts an exact tool count or an exact name set for
the seed set, update it to include the nine — that assertion is now describing
the intended new behavior, not catching a regression.

- [ ] **Step 7: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/tools/public_data/__init__.py src/internal/tools/knowledge_base.py tests/unit/test_public_data_seeding.py
git commit -m "feat(tools): seed the nine public data tools as agent-callable"
```

---

### Task 6: Source cards from citeable tools, plus docs

**Files:**
- Modify: `src/internal/servers/web/tool_agent_runner.py:56` (signature),
  `:97-110` (doc building), `:190` (call site)
- Modify: `tests/unit/test_tool_agent_runner.py`
- Modify: `README.md`, `.claude/CLAUDE.md`

**Interfaces:**
- Consumes from Task 5: the seeded citeable tools.
- Produces: `_extract_tool_calls_and_docs(output, citeable_tools=frozenset({_CORPUS_SEARCH_NAME}))`.
  The default preserves today's behavior for existing callers.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_tool_agent_runner.py`:

```python
def test_citeable_tool_output_becomes_documents():
    """A citeable tool's {title, content, url} array turns into source cards."""
    import json

    from src.internal.servers.web.tool_agent_runner import (
        _extract_tool_calls_and_docs,
    )

    class _Out:
        action_trace = json.dumps(
            {
                "tool_name": "search_wikipedia",
                "status": "completed",
                "arguments": {"query": "faiss"},
                "result": json.dumps(
                    [{"title": "FAISS", "content": "A library.", "url": "u1"}]
                ),
            }
        )

    _calls, docs = _extract_tool_calls_and_docs(
        _Out(), frozenset({"search_wikipedia"})
    )

    assert len(docs) == 1
    assert docs[0].title == "FAISS"
    assert docs[0].metadata["source"] == "search_wikipedia"


def test_non_citeable_tool_output_is_not_cited():
    import json

    from src.internal.servers.web.tool_agent_runner import (
        _extract_tool_calls_and_docs,
    )

    class _Out:
        action_trace = json.dumps(
            {
                "tool_name": "get_weather",
                "status": "completed",
                "arguments": {"location": "Berlin"},
                "result": json.dumps({"temperature": 14.2}),
            }
        )

    _calls, docs = _extract_tool_calls_and_docs(
        _Out(), frozenset({"search_wikipedia"})
    )

    assert docs == []


def test_document_ids_are_unique_across_citeable_tools():
    """Two citeable tools in one turn must not both emit D1."""
    import json

    from src.internal.servers.web.tool_agent_runner import (
        _extract_tool_calls_and_docs,
    )

    def _record(name, url):
        return json.dumps(
            {
                "tool_name": name,
                "status": "completed",
                "arguments": {},
                "result": json.dumps([{"title": name, "content": "c", "url": url}]),
            }
        )

    class _Out:
        action_trace = "\n".join(
            [_record("search_wikipedia", "u1"), _record("search_arxiv", "u2")]
        )

    _calls, docs = _extract_tool_calls_and_docs(
        _Out(), frozenset({"search_wikipedia", "search_arxiv"})
    )

    assert [d.id for d in docs] == ["D1", "D2"]


def test_citeable_tool_with_non_conforming_result_yields_no_documents():
    """web_search is citeable but answers with prose; that must not error."""
    import json

    from src.internal.servers.web.tool_agent_runner import (
        _extract_tool_calls_and_docs,
    )

    class _Out:
        action_trace = json.dumps(
            {
                "tool_name": "web_search",
                "status": "completed",
                "arguments": {},
                "result": "1. Some Page\nA prose summary.",
            }
        )

    calls, docs = _extract_tool_calls_and_docs(_Out(), frozenset({"web_search"}))

    assert docs == []
    assert calls[0].tool_name == "web_search"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_tool_agent_runner.py -v`
Expected: FAIL — `_extract_tool_calls_and_docs() takes 1 positional argument but 2 were given`

- [ ] **Step 3: Generalize the extractor**

In `src/internal/servers/web/tool_agent_runner.py`, change the signature at
line 56 from:

```python
def _extract_tool_calls_and_docs(output) -> tuple[list, list]:
    """Parse a ToolAgentLoop action_trace into ToolCallView + ContextDocument lists."""
```

to:

```python
def _extract_tool_calls_and_docs(
    output,
    citeable_tools: frozenset[str] = frozenset({_CORPUS_SEARCH_NAME}),
) -> tuple[list, list]:
    """Parse a ToolAgentLoop action_trace into ToolCallView + ContextDocument lists.

    *citeable_tools* names the tools whose results become source cards. The
    caller derives it from ``tool.citeable`` on the tools it passed to the loop,
    so renaming a tool cannot silently drop its citations. The default keeps the
    corpus-search-only behavior for callers that do not pass a set.
    """
```

Then replace the doc-building block (currently lines 97-110):

```python
            if tool_name == _CORPUS_SEARCH_NAME and result:
                raw = _json.loads(result) if isinstance(result, str) else result
                if isinstance(raw, list):
                    for i, item in enumerate(raw, 1):
                        documents.append(
                            ContextDocument(
                                id=f"D{i}",
                                title=item.get("title", ""),
                                content=item.get("content", ""),
                                url=item.get("url"),
                                score=0.0,
                                metadata={"source": _CORPUS_SEARCH_NAME},
                            )
                        )
```

with:

```python
            if tool_name in citeable_tools and result:
                # Citeable tools answer with a JSON array of
                # {title, content, url}. A citeable tool that answers some
                # other way (web_search returns prose) simply contributes no
                # cards rather than failing the turn.
                raw = decoded_result
                if isinstance(raw, list):
                    for item in raw:
                        if not isinstance(item, dict):
                            continue
                        documents.append(
                            ContextDocument(
                                # Numbered across the whole trace: several
                                # citeable tools can run in one turn, and
                                # restarting per tool would emit two D1s.
                                id=f"D{len(documents) + 1}",
                                title=item.get("title", ""),
                                content=item.get("content", ""),
                                url=item.get("url"),
                                score=0.0,
                                metadata={"source": tool_name},
                            )
                        )
```

Note this reuses `decoded_result`, which the loop already computed above, so
the old redundant `_json.loads(result)` goes away.

- [ ] **Step 4: Pass the real citeable set at the call site**

In the same file, immediately after the `if with_search_tool:` block that
builds `tools` (just before `loop = ToolAgentLoop(`), add:

```python
    # Derived from the tools actually offered this turn, so a rename or a
    # withheld tool cannot leave a stale name behind.
    citeable_tool_names = frozenset(t.name for t in tools if t.citeable)
```

Then change line 190 from:

```python
    tool_calls, documents = _extract_tool_calls_and_docs(output)
```

to:

```python
    tool_calls, documents = _extract_tool_calls_and_docs(output, citeable_tool_names)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/unit/test_tool_agent_runner.py -v`
Expected: PASS, including the pre-existing tests that call
`_extract_tool_calls_and_docs(output)` with one argument.

- [ ] **Step 6: Run the whole suite**

Run: `pytest`
Expected: PASS. Investigate any failure before continuing; do not adjust an
assertion unless it is genuinely describing the old seed set.

- [ ] **Step 7: Update the docs**

In `README.md`, find the section describing the tool framework or the `/tools`
surface and add:

```markdown
#### Built-in public data tools

The tool agent ships nine keyless public data-source tools, seeded by
`src/internal/tools/public_data/`. They need no API keys or configuration:

| Tool | Source |
| --- | --- |
| `search_wikipedia` | Wikipedia action API |
| `search_arxiv` | ArXiv export API |
| `search_wayback` | Internet Archive CDX API |
| `get_weather` | Open-Meteo |
| `get_stock_quote` | Yahoo Finance chart API |
| `get_crypto_price` | CoinGecko |
| `convert_currency` | exchangerate-api.com |
| `search_location` | Nominatim (OpenStreetMap) |
| `search_nearby_places` | Overpass (OpenStreetMap) |

The first three are citeable: they answer with `{title, content, url}` records,
so their results appear as source cards on `/tools`. The rest answer with a
JSON object of facts. Any upstream failure returns `{"error": ...}` from that
one tool and leaves the turn intact.
```

In `.claude/CLAUDE.md`, under the Architecture section's description of the
tool framework, add one line:

```markdown
- `src/internal/tools/public_data/` — nine keyless public data-source tools
  (Wikipedia, ArXiv, Wayback, weather, stocks, crypto, currency, geocoding,
  nearby places), seeded into the registry by `tool_knowledge_base()`
```

- [ ] **Step 8: Verify no dependency crept in**

Run: `git diff origin/main -- requirements.txt`
Expected: empty output.

- [ ] **Step 9: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/servers/web/tool_agent_runner.py tests/unit/test_tool_agent_runner.py README.md .claude/CLAUDE.md
git commit -m "feat(tools): render citeable tool results as source cards"
```

---

## Manual verification

After Task 6, confirm the surface actually works end to end:

1. Start the retrieval server and the web backend per `.claude/CLAUDE.md`.
2. `curl -s localhost:7860/api/debug/tools | python3 -m json.tool | grep -c search_wikipedia`
   — expect a non-zero count. (Requires `AGENTIC_SEARCH_DEBUG_PANELS`; the
   debug router only mounts under that flag.)
3. Open `http://127.0.0.1:5173/tools` and ask "What is the weather in Berlin?"
   — the trace should show a `get_weather` call.
4. Ask "Find me ArXiv papers about dense retrieval" — the trace should show
   `search_arxiv` and the answer should carry source cards.

Tool selection quality on a small local model is the open risk the spec
accepted. If the model calls the wrong tool in step 3 or 4, record what it
picked; that is the evidence that would justify revisiting the
"offer all nine" decision.
