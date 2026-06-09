# Multi-Provider Web Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `src/tools/search.py` with patterns borrowed from Onyx reference code: query sanitization, Brave Search provider, Serper.dev provider, and a `MultiQueryWebSearchTool(Tool)` class that accepts a list of queries and runs them in parallel.

**Architecture:** All additions go into `src/tools/search.py` (existing module). New functions follow the exact same pattern as `google_custom_search`/`serpapi_search`. The `MultiQueryWebSearchTool` subclasses `base.py`'s `Tool` ABC (`async execute(instance_id, args) -> tuple[str, Any, Any]`). Nothing touches the agent loops — these are drop-in additions to the existing module.

**Tech Stack:** Python 3.12, aiohttp (already imported via `..context.retrieval.client`), pytest, `unittest.mock`

**Onyx reference:** Commit `02c73a3` in this repo — `src/tools/tool_implementations/web_search/web_search_tool.py` (sanitization, multi-query) and `clients/brave_client.py` / `clients/serper_client.py` (provider implementations).

---

## File Map

| File | Change |
|---|---|
| `src/tools/search.py` | Add `_sanitize_query`, `_normalize_queries_input`, `brave_search`, `serper_dev_search`, extend `SearchProvider` literal, update `search_tool` dispatch, add `MultiQueryWebSearchTool` |
| `src/tools/__init__.py` | Export `brave_search`, `serper_dev_search`, `MultiQueryWebSearchTool` |
| `tests/unit/test_search_tools.py` | Add tests for all new functions and the Tool class |

---

## Task 1: Add query sanitization utilities

**Files:**
- Modify: `src/tools/search.py` (add two private functions near the top)
- Modify: `tests/unit/test_search_tools.py` (add test class)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_search_tools.py`:

```python
from src.tools.search import _normalize_queries_input, _sanitize_query


class TestSanitizeQuery:
    def test_removes_null_bytes(self):
        assert _sanitize_query("hello\x00world") == "hello world"

    def test_removes_control_chars(self):
        assert _sanitize_query("a\x01b\x1fc") == "a b c"

    def test_removes_del_char(self):
        assert _sanitize_query("abc\x7fdef") == "abcdef"

    def test_normalizes_whitespace(self):
        assert _sanitize_query("  foo   bar  ") == "foo bar"

    def test_passthrough_clean_query(self):
        assert _sanitize_query("What is FAISS?") == "What is FAISS?"


class TestNormalizeQueriesInput:
    def test_string_becomes_list(self):
        assert _normalize_queries_input("hello") == ["hello"]

    def test_list_passthrough(self):
        assert _normalize_queries_input(["a", "b"]) == ["a", "b"]

    def test_drops_empty_strings(self):
        assert _normalize_queries_input(["ok", "", "  "]) == ["ok"]

    def test_non_list_non_string_returns_empty(self):
        assert _normalize_queries_input(42) == []
        assert _normalize_queries_input(None) == []

    def test_sanitizes_each_entry(self):
        assert _normalize_queries_input(["hello\x00world"]) == ["hello world"]

    def test_none_items_dropped(self):
        assert _normalize_queries_input(["a", None, "b"]) == ["a", "b"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/linghuang/Git/Agentic-Search && pytest tests/unit/test_search_tools.py::TestSanitizeQuery tests/unit/test_search_tools.py::TestNormalizeQueriesInput -v --tb=short 2>&1 | tail -20
```

Expected: ImportError or AttributeError (functions don't exist yet).

- [ ] **Step 3: Implement the functions**

In `src/tools/search.py`, after the `DEFAULT_USER_AGENT` constant (around line 26), add:

```python
def _sanitize_query(query: str) -> str:
    sanitized = "".join(c for c in query if ord(c) >= 32 and ord(c) != 127)
    return " ".join(sanitized.split())


def _normalize_queries_input(raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        raw = [raw]
    elif not isinstance(raw, list):
        return []
    result: list[str] = []
    for q in raw:
        if q is None:
            continue
        sanitized = _sanitize_query(str(q))
        if sanitized:
            result.append(sanitized)
    return result
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/linghuang/Git/Agentic-Search && pytest tests/unit/test_search_tools.py::TestSanitizeQuery tests/unit/test_search_tools.py::TestNormalizeQueriesInput -v --tb=short 2>&1 | tail -20
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tools/search.py tests/unit/test_search_tools.py
git commit -m "$(cat <<'EOF'
feat: add query sanitization utilities to search module

Borrowed from Onyx reference: _sanitize_query strips control characters
and normalizes whitespace; _normalize_queries_input accepts string or list
from LLM and sanitizes each entry.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add Brave Search provider

**Files:**
- Modify: `src/tools/search.py`
- Modify: `tests/unit/test_search_tools.py`

Brave API docs: POST `https://api.search.brave.com/res/v1/web/search` with header `X-Subscription-Token: <api_key>` and query params `q`, `count`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_search_tools.py`:

```python
from src.tools.search import brave_search


class TestBraveSearch:
    def test_returns_mapped_results(self, monkeypatch):
        import src.tools.search as mod

        payload = {
            "web": {
                "results": [
                    {"title": "BraveOne", "url": "https://brave.test/1", "description": "desc1"},
                    {"title": "BraveTwo", "url": "https://brave.test/2", "description": "desc2"},
                ]
            }
        }
        monkeypatch.setattr(mod, "aiohttp", _make_fake_aiohttp(payload))

        pages = asyncio.get_event_loop().run_until_complete(
            brave_search("test query", api_key="fake-key")
        )
        assert len(pages) == 2
        assert pages[0].title == "BraveOne"
        assert pages[0].url == "https://brave.test/1"
        assert pages[0].summary == "desc1"

    def test_returns_error_page_on_missing_api_key(self, monkeypatch):
        import os
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)

        pages = asyncio.get_event_loop().run_until_complete(
            brave_search("test", api_key=None)
        )
        assert len(pages) == 1
        assert pages[0].error is not None

    def test_returns_error_page_on_http_failure(self, monkeypatch):
        import src.tools.search as mod
        monkeypatch.setattr(mod, "aiohttp", _make_fake_aiohttp_error())

        pages = asyncio.get_event_loop().run_until_complete(
            brave_search("test", api_key="k")
        )
        assert len(pages) == 1
        assert pages[0].error is not None
```

Add helpers at the top of the test file (after existing `_FakeSession`):

```python
def _make_fake_aiohttp(payload):
    import types, aiohttp as real_aiohttp

    class FakeResp:
        def __init__(self): self.headers = {"content-type": "application/json"}
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def raise_for_status(self): pass
        async def json(self): return payload
        async def text(self): return ""

    class FakeSess:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def get(self, url, **kw): return FakeResp()
        def post(self, url, **kw): return FakeResp()

    fake = types.SimpleNamespace(
        ClientSession=FakeSess,
        ClientTimeout=real_aiohttp.ClientTimeout,
    )
    return fake


def _make_fake_aiohttp_error():
    import types, aiohttp as real_aiohttp

    class FakeResp:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def raise_for_status(self): raise Exception("HTTP error")

    class FakeSess:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def get(self, url, **kw): return FakeResp()

    return types.SimpleNamespace(
        ClientSession=FakeSess,
        ClientTimeout=real_aiohttp.ClientTimeout,
    )
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/linghuang/Git/Agentic-Search && pytest tests/unit/test_search_tools.py::TestBraveSearch -v --tb=short 2>&1 | tail -20
```

Expected: ImportError for `brave_search`.

- [ ] **Step 3: Implement `brave_search`**

In `src/tools/search.py`, update `SearchProvider` literal and add `BRAVE_SEARCH_ENDPOINT` constant, then add the function after `serpapi_search`:

```python
BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
```

Update `SearchProvider` (top of file):
```python
SearchProvider = Literal["retrieval", "google", "serpapi", "brave", "serper"]
```

Add function after `serpapi_search`:

```python
async def brave_search(
    query: str,
    *,
    page_size: int = 5,
    api_key: str | None = None,
    timeout_seconds: int = 10,
) -> list[SearchPage]:
    """Search Brave Search API and return normalized pages."""

    api_key = api_key or os.getenv("BRAVE_API_KEY")
    if not api_key:
        return [SearchPage(error="BRAVE_API_KEY is required.")]

    try:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        }
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                BRAVE_SEARCH_ENDPOINT,
                params={"q": query, "count": page_size},
                headers=headers,
            ) as response:
                response.raise_for_status()
                data = await response.json()
    except Exception as exc:
        return [SearchPage(error=_redact_secret_params(str(exc)))]

    results = (data.get("web") or {}).get("results") or []
    return [
        SearchPage(
            title=item.get("title", ""),
            summary=item.get("description", ""),
            url=item.get("url", ""),
        )
        for item in results[:page_size]
    ]
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/linghuang/Git/Agentic-Search && pytest tests/unit/test_search_tools.py::TestBraveSearch -v --tb=short 2>&1 | tail -20
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tools/search.py tests/unit/test_search_tools.py
git commit -m "$(cat <<'EOF'
feat: add Brave Search provider to search module

New brave_search() follows the same pattern as google_custom_search()
and serpapi_search(). Reads BRAVE_API_KEY from env if not passed directly.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add Serper.dev provider

**Files:**
- Modify: `src/tools/search.py`
- Modify: `tests/unit/test_search_tools.py`

Serper is a different service from SerpAPI. Endpoint: `POST https://google.serper.dev/search` with header `X-API-KEY: <key>` and JSON body `{"q": ..., "num": ...}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_search_tools.py`:

```python
from src.tools.search import serper_dev_search


class TestSerperDevSearch:
    def test_returns_mapped_results(self, monkeypatch):
        import src.tools.search as mod

        payload = {
            "organic": [
                {"title": "SerperOne", "link": "https://serper.test/1", "snippet": "snip1"},
            ]
        }

        class FakePostResp:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            def raise_for_status(self): pass
            async def json(self): return payload

        class FakeSess:
            def __init__(self, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            def post(self, url, **kw): return FakePostResp()

        import types
        monkeypatch.setattr(
            mod, "aiohttp",
            types.SimpleNamespace(ClientSession=FakeSess, ClientTimeout=mod.aiohttp.ClientTimeout)
        )

        pages = asyncio.get_event_loop().run_until_complete(
            serper_dev_search("test", api_key="key")
        )
        assert len(pages) == 1
        assert pages[0].title == "SerperOne"
        assert pages[0].url == "https://serper.test/1"

    def test_returns_error_on_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        pages = asyncio.get_event_loop().run_until_complete(
            serper_dev_search("test", api_key=None)
        )
        assert pages[0].error is not None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/linghuang/Git/Agentic-Search && pytest tests/unit/test_search_tools.py::TestSerperDevSearch -v --tb=short 2>&1 | tail -20
```

Expected: ImportError for `serper_dev_search`.

- [ ] **Step 3: Implement `serper_dev_search`**

Add constant near the other endpoint constants in `src/tools/search.py`:

```python
SERPER_DEV_ENDPOINT = "https://google.serper.dev/search"
```

Add function after `brave_search`:

```python
async def serper_dev_search(
    query: str,
    *,
    page_size: int = 5,
    api_key: str | None = None,
    timeout_seconds: int = 10,
) -> list[SearchPage]:
    """Search via Serper.dev (Google results) and return normalized pages."""

    api_key = api_key or os.getenv("SERPER_API_KEY")
    if not api_key:
        return [SearchPage(error="SERPER_API_KEY is required.")]

    try:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                SERPER_DEV_ENDPOINT,
                json={"q": query, "num": page_size},
                headers=headers,
            ) as response:
                response.raise_for_status()
                data = await response.json()
    except Exception as exc:
        return [SearchPage(error=_redact_secret_params(str(exc)))]

    results = data.get("organic") or []
    return [
        SearchPage(
            title=item.get("title", ""),
            summary=item.get("snippet", ""),
            url=item.get("link", ""),
        )
        for item in results[:page_size]
    ]
```

Also update `search_tool` dispatch to handle the two new providers. Find the `search_tool` function and add branches:

```python
    if provider == "brave":
        return await brave_search(
            query,
            page_size=page_size,
            timeout_seconds=timeout_seconds,
        )
    if provider == "serper":
        return await serper_dev_search(
            query,
            page_size=page_size,
            timeout_seconds=timeout_seconds,
        )
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/linghuang/Git/Agentic-Search && pytest tests/unit/test_search_tools.py::TestSerperDevSearch -v --tb=short 2>&1 | tail -20
```

Expected: all 2 PASS.

- [ ] **Step 5: Run the full existing test suite to check nothing broke**

```bash
cd /Users/linghuang/Git/Agentic-Search && pytest tests/unit/test_search_tools.py -v --tb=short 2>&1 | tail -30
```

Expected: all pre-existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/tools/search.py tests/unit/test_search_tools.py
git commit -m "$(cat <<'EOF'
feat: add Serper.dev provider and wire new providers into search_tool

serper_dev_search() calls serper.dev (distinct from the existing
serpapi_search which calls serpapi.com). SearchProvider literal extended
to include 'brave' and 'serper'.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add `MultiQueryWebSearchTool(Tool)` class

**Files:**
- Modify: `src/tools/search.py`
- Modify: `tests/unit/test_search_tools.py`

This is a proper `Tool` ABC subclass (using `base.py`'s interface) that accepts `{"queries": ["q1", "q2"]}` from the LLM, sanitizes queries with `_normalize_queries_input`, runs them all in parallel with `asyncio.gather`, and returns merged deduplicated results. This enables `ToolAgentLoop` to issue multiple parallel searches per turn.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_search_tools.py`:

```python
from src.tools.search import MultiQueryWebSearchTool


class TestMultiQueryWebSearchTool:
    def _make_tool(self, pages_per_query=None):
        """Build a MultiQueryWebSearchTool with patched search_tool."""
        if pages_per_query is None:
            pages_per_query = [SearchPage(title="T", summary="S", url="https://t.test")]

        async def _fake_search_tool(query, **kwargs):
            return pages_per_query

        return MultiQueryWebSearchTool(search_fn=_fake_search_tool)

    def test_schema_has_queries_field(self):
        tool = self._make_tool()
        schema = tool.schema
        assert schema.name == "web_search"
        props = schema.parameters["properties"]
        assert "queries" in props
        assert props["queries"]["type"] == "array"

    def test_execute_runs_queries_in_parallel(self):
        seen = []

        async def _fake(query, **kwargs):
            seen.append(query)
            return [SearchPage(title=query, summary="s", url=f"https://{query}.test")]

        tool = MultiQueryWebSearchTool(search_fn=_fake)
        result_str, raw, meta = asyncio.get_event_loop().run_until_complete(
            tool.execute("inst1", {"queries": ["alpha", "beta"]})
        )
        assert "alpha" in result_str
        assert "beta" in result_str
        assert set(seen) == {"alpha", "beta"}

    def test_execute_deduplicates_by_url(self):
        async def _fake(query, **kwargs):
            return [SearchPage(title="Same", summary="s", url="https://same.test")]

        tool = MultiQueryWebSearchTool(search_fn=_fake)
        result_str, raw, _ = asyncio.get_event_loop().run_until_complete(
            tool.execute("inst1", {"queries": ["q1", "q2"]})
        )
        assert result_str.count("https://same.test") == 1

    def test_execute_sanitizes_queries(self):
        seen = []

        async def _fake(query, **kwargs):
            seen.append(query)
            return []

        tool = MultiQueryWebSearchTool(search_fn=_fake)
        asyncio.get_event_loop().run_until_complete(
            tool.execute("inst1", {"queries": ["hello\x00world", "  ok  "]})
        )
        assert seen == ["hello world", "ok"]

    def test_execute_returns_no_results_string_when_empty(self):
        async def _fake(query, **kwargs):
            return []

        tool = MultiQueryWebSearchTool(search_fn=_fake)
        result_str, raw, _ = asyncio.get_event_loop().run_until_complete(
            tool.execute("inst1", {"queries": ["nothing"]})
        )
        assert result_str == "No results found."

    def test_execute_accepts_string_queries(self):
        seen = []

        async def _fake(query, **kwargs):
            seen.append(query)
            return []

        tool = MultiQueryWebSearchTool(search_fn=_fake)
        asyncio.get_event_loop().run_until_complete(
            tool.execute("inst1", {"queries": "single query"})
        )
        assert seen == ["single query"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/linghuang/Git/Agentic-Search && pytest tests/unit/test_search_tools.py::TestMultiQueryWebSearchTool -v --tb=short 2>&1 | tail -20
```

Expected: ImportError for `MultiQueryWebSearchTool`.

- [ ] **Step 3: Implement `MultiQueryWebSearchTool`**

Add this class at the bottom of `src/tools/search.py`, before the private helpers section. It imports `Tool` and `ToolSchema` from `.base`:

```python
from .base import FunctionTool, Tool, ToolSchema
```

(The file already imports `FunctionTool`; just extend the import.)

Then add the class:

```python
class MultiQueryWebSearchTool(Tool):
    """Tool that accepts multiple queries and runs them in parallel.

    Designed for use with ToolAgentLoop. The LLM passes {"queries": ["q1", "q2"]}
    and all queries execute concurrently, with results deduplicated by URL.
    """

    def __init__(
        self,
        search_fn: Any = None,
        *,
        provider: SearchProvider = "retrieval",
        search_url: str = "http://localhost:8000/retrieve",
        page_size: int = 5,
        timeout_seconds: int = 15,
    ) -> None:
        self._search_fn = search_fn or search_tool
        self._provider = provider
        self._search_url = search_url
        self._page_size = page_size
        self._timeout_seconds = timeout_seconds
        self._schema = ToolSchema(
            name="web_search",
            description=(
                "Search the web for information. Pass multiple queries to search in parallel."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "One or more search queries to run in parallel.",
                    }
                },
                "required": ["queries"],
            },
        )

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    async def execute(
        self, instance_id: str, arguments: dict[str, Any]
    ) -> tuple[str, Any, Any]:
        del instance_id
        raw_queries = arguments.get("queries", [])
        queries = _normalize_queries_input(raw_queries)

        if not queries:
            return "No results found.", [], {}

        results_per_query: list[list[SearchPage]] = await asyncio.gather(
            *[
                self._search_fn(
                    q,
                    provider=self._provider,
                    search_url=self._search_url,
                    page_size=self._page_size,
                    timeout_seconds=self._timeout_seconds,
                )
                for q in queries
            ]
        )

        seen_urls: set[str] = set()
        merged: list[SearchPage] = []
        for pages in results_per_query:
            for page in pages:
                if page.url and page.url in seen_urls:
                    continue
                if page.url:
                    seen_urls.add(page.url)
                merged.append(page)

        return format_search_pages(merged), merged, {"queries": queries}
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/linghuang/Git/Agentic-Search && pytest tests/unit/test_search_tools.py::TestMultiQueryWebSearchTool -v --tb=short 2>&1 | tail -20
```

Expected: all 6 PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
cd /Users/linghuang/Git/Agentic-Search && pytest tests/unit/ -v --tb=short 2>&1 | tail -30
```

Expected: all pre-existing tests pass plus new ones.

- [ ] **Step 6: Commit**

```bash
git add src/tools/search.py tests/unit/test_search_tools.py
git commit -m "$(cat <<'EOF'
feat: add MultiQueryWebSearchTool for parallel multi-query search

Tool(ABC) subclass accepting {"queries": [...]} — sanitizes with
_normalize_queries_input, runs all queries in parallel via asyncio.gather,
deduplicates results by URL. Compatible with ToolAgentLoop.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Export new symbols from `src/tools/__init__.py`

**Files:**
- Modify: `src/tools/__init__.py`

- [ ] **Step 1: Add exports**

Current `src/tools/__init__.py` ends at line 20. Append the new exports:

```python
from .search import MultiQueryWebSearchTool as MultiQueryWebSearchTool
from .search import brave_search as brave_search
from .search import serper_dev_search as serper_dev_search
```

- [ ] **Step 2: Verify imports work**

```bash
cd /Users/linghuang/Git/Agentic-Search && python3 -c "from src.tools import MultiQueryWebSearchTool, brave_search, serper_dev_search; print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Run full unit tests one final time**

```bash
cd /Users/linghuang/Git/Agentic-Search && pytest tests/unit/ --tb=short 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/tools/__init__.py
git commit -m "$(cat <<'EOF'
feat: export new search symbols via src/tools package interface

Adds MultiQueryWebSearchTool, brave_search, serper_dev_search to the
package's public API.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```
