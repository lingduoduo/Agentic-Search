# Optimize Web Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich web search results with full page content fetched concurrently from source URLs, and inject temporal context for time-sensitive queries, so the system returns up-to-date information instead of stale API snippets.

**Architecture:** After `serpapi` and `google` searches return title+URL+snippet, a concurrent page fetcher visits each URL, extracts article-quality text (preferring `<article>`/`<main>` elements), and replaces the snippet. For time-sensitive queries (containing "latest", "current", "today", etc.), a temporal variant appending the current month/year is added to the expanded query list so the search engine prefers recent results.

**Tech Stack:** Python `asyncio` (concurrent fetching, already used), `BeautifulSoup` (HTML extraction, already imported), `aiohttp` (HTTP client, already imported), `pytest` + `monkeypatch` for tests.

---

## File map

| File | Change |
|---|---|
| `src/tools/search.py` | Improve `_html_to_text()` extraction; add `fetch_pages_concurrently()`; bump `fetch_url()` default limit |
| `src/backend/secondary_llm_flows/query_expansion.py` | Add `is_time_sensitive()` + `with_temporal_context()` |
| `src/backend/servers/web/app.py` | Wire content fetching into `_run_direct_search()` and `_run_hybrid_search()` for web providers |
| `tests/unit/test_search_tools.py` | Extend with `fetch_pages_concurrently` tests |
| `tests/unit/test_secondary_llm_flows.py` | Extend with temporal context tests |

---

### Task 1: Improve HTML extraction and add concurrent page fetcher

**Files:**
- Modify: `src/tools/search.py`
- Test: `tests/unit/test_search_tools.py`

- [ ] **Step 1: Write the failing tests**

Add to the bottom of `tests/unit/test_search_tools.py`:

```python
from src.tools.search import fetch_pages_concurrently


def test_html_to_text_prefers_article_element():
    """_html_to_text should extract <article> content before falling back to <p> tags."""
    from src.tools.search import _html_to_text

    html = """
    <html><body>
      <header><p>Navigation noise</p></header>
      <article><p>Main article paragraph one.</p><p>Paragraph two.</p></article>
      <footer><p>Footer noise</p></footer>
    </body></html>
    """
    text = _html_to_text(html)
    assert "Main article paragraph one." in text
    # Should NOT include noise from header/footer when article has enough content
    assert "Navigation noise" not in text


def test_html_to_text_falls_back_to_p_tags_when_no_article():
    from src.tools.search import _html_to_text

    html = "<html><body><div><p>Content here.</p><p>More content.</p></div></body></html>"
    text = _html_to_text(html)
    assert "Content here." in text
    assert "More content." in text


def test_fetch_pages_concurrently_replaces_summary_with_fetched_content(monkeypatch):
    async def _fake_fetch_url(url, *, max_length, timeout_seconds):
        return f"fetched:{url}"

    monkeypatch.setattr("src.tools.search.fetch_url", _fake_fetch_url)

    pages = [
        SearchPage(title="A", summary="short", url="https://a.test"),
        SearchPage(title="B", summary="short", url="https://b.test"),
    ]
    enriched = asyncio.run(fetch_pages_concurrently(pages, max_chars=2000))

    assert enriched[0].summary == "fetched:https://a.test"
    assert enriched[1].summary == "fetched:https://b.test"
    assert enriched[0].title == "A"  # title preserved


def test_fetch_pages_concurrently_skips_error_and_empty_url_pages(monkeypatch):
    async def _fake_fetch_url(url, **kwargs):
        return "fetched"

    monkeypatch.setattr("src.tools.search.fetch_url", _fake_fetch_url)

    pages = [
        SearchPage(error="oops"),
        SearchPage(title="NoURL", summary="s", url=""),
        SearchPage(title="OK", summary="s", url="https://ok.test"),
    ]
    enriched = asyncio.run(fetch_pages_concurrently(pages, max_chars=2000))

    assert enriched[0].error == "oops"    # error page unchanged
    assert enriched[1].summary == "s"     # no-URL page unchanged
    assert enriched[2].summary == "fetched"


def test_fetch_pages_concurrently_keeps_original_on_fetch_error(monkeypatch):
    async def _fake_fetch_url(url, **kwargs):
        return "[fetch error] timeout"

    monkeypatch.setattr("src.tools.search.fetch_url", _fake_fetch_url)

    pages = [SearchPage(title="T", summary="original", url="https://t.test")]
    enriched = asyncio.run(fetch_pages_concurrently(pages, max_chars=2000))

    assert enriched[0].summary == "original"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_search_tools.py::test_html_to_text_prefers_article_element \
       tests/unit/test_search_tools.py::test_fetch_pages_concurrently_replaces_summary_with_fetched_content \
       -v
```

Expected: `FAILED` — `fetch_pages_concurrently` does not exist yet.

- [ ] **Step 3: Implement the changes in `src/tools/search.py`**

Replace the existing `_html_to_text` function (lines 402–413) and `fetch_url` function (lines 254–270) with:

```python
async def fetch_url(
    url: str, *, max_length: int = 2000, timeout_seconds: int = 15
) -> str:
    """Fetch readable webpage text with lightweight HTML extraction."""

    if not url:
        return ""
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        headers = {"User-Agent": DEFAULT_USER_AGENT}
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                html = await response.text()
        return _html_to_text(html)[:max_length]
    except Exception as exc:
        return f"[fetch error] {exc}"


async def fetch_pages_concurrently(
    pages: list[SearchPage],
    *,
    max_chars: int = 2000,
    timeout_seconds: int = 10,
) -> list[SearchPage]:
    """Fetch full page content for each SearchPage that has a URL and no error."""
    fetchable = [p for p in pages if p.url and not p.error]
    results = await asyncio.gather(
        *[
            fetch_url(p.url, max_length=max_chars, timeout_seconds=timeout_seconds)
            for p in fetchable
        ],
        return_exceptions=True,
    )
    url_to_content: dict[str, str] = {}
    for page, content in zip(fetchable, results):
        if isinstance(content, str) and not content.startswith("[fetch error]"):
            url_to_content[page.url] = content

    return [
        SearchPage(
            title=p.title,
            summary=url_to_content.get(p.url, p.summary) if (p.url and not p.error) else p.summary,
            url=p.url,
            error=p.error,
        )
        for p in pages
    ]
```

Also replace `_html_to_text` (lines 402–413):

```python
def _html_to_text(html: str) -> str:
    try:
        import bs4
    except ImportError:
        return " ".join(html.split())

    soup = bs4.BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Prefer semantic content containers before falling back to all <p> tags
    for selector in ("article", "main", '[role="main"]', ".content", "#content"):
        container = soup.select_one(selector)
        if container:
            paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
            text = "\n".join(p for p in paragraphs if p)
            if len(text) > 200:
                return text

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join(paragraph for paragraph in paragraphs if paragraph)
    return text or soup.get_text(" ", strip=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_search_tools.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tools/search.py tests/unit/test_search_tools.py
git commit -m "feat: add fetch_pages_concurrently and improve HTML extraction in search tools"
```

---

### Task 2: Add temporal query context injection

**Files:**
- Modify: `src/backend/secondary_llm_flows/query_expansion.py`
- Test: `tests/unit/test_secondary_llm_flows.py`

- [ ] **Step 1: Write the failing tests**

Add to the bottom of `tests/unit/test_secondary_llm_flows.py`:

```python
from src.backend.secondary_llm_flows.query_expansion import (
    is_time_sensitive,
    with_temporal_context,
)


def test_is_time_sensitive_detects_temporal_keywords():
    assert is_time_sensitive("latest AI news") is True
    assert is_time_sensitive("current president of France") is True
    assert is_time_sensitive("recent breakthroughs in quantum computing") is True
    assert is_time_sensitive("today's stock price") is True
    assert is_time_sensitive("what is FAISS") is False
    assert is_time_sensitive("how does BM25 work") is False


def test_is_time_sensitive_is_case_insensitive():
    assert is_time_sensitive("LATEST updates") is True
    assert is_time_sensitive("Current events") is True


def test_with_temporal_context_appends_date_to_time_sensitive_queries():
    from datetime import datetime

    result = with_temporal_context("latest AI models")
    year = str(datetime.now().year)
    assert year in result
    assert "latest AI models" in result


def test_with_temporal_context_leaves_non_temporal_queries_unchanged():
    result = with_temporal_context("what is FAISS")
    assert result == "what is FAISS"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_secondary_llm_flows.py::test_is_time_sensitive_detects_temporal_keywords \
       tests/unit/test_secondary_llm_flows.py::test_with_temporal_context_appends_date_to_time_sensitive_queries \
       -v
```

Expected: `FAILED` — `is_time_sensitive` and `with_temporal_context` do not exist yet.

- [ ] **Step 3: Implement in `src/backend/secondary_llm_flows/query_expansion.py`**

Add after the existing imports at the top:

```python
from datetime import datetime
```

Add after the `_LIST_MARKER_RE` constant (line 17):

```python
_TEMPORAL_KEYWORDS = frozenset(
    {
        "latest",
        "recent",
        "current",
        "today",
        "now",
        "this year",
        "this month",
        "newest",
        "new",
        "updated",
        "2024",
        "2025",
        "2026",
    }
)


def is_time_sensitive(query: str) -> bool:
    """Return True when the query contains temporal keywords that suggest recency matters."""
    lower = query.lower()
    return any(kw in lower for kw in _TEMPORAL_KEYWORDS)


def with_temporal_context(query: str) -> str:
    """Append current month and year to time-sensitive queries; return query unchanged otherwise."""
    if not is_time_sensitive(query):
        return query
    date_str = datetime.now().strftime("%B %Y")
    return f"{query} {date_str}"
```

Update the `__all__` at the bottom of the file:

```python
__all__ = ["expand_keywords", "is_time_sensitive", "with_temporal_context"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_secondary_llm_flows.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backend/secondary_llm_flows/query_expansion.py tests/unit/test_secondary_llm_flows.py
git commit -m "feat: add temporal query context detection and injection"
```

---

### Task 3: Wire content fetching into direct search mode

**Files:**
- Modify: `src/backend/servers/web/app.py`
- Test: `tests/unit/servers/web/test_web_experience_app.py`

- [ ] **Step 1: Write the failing test**

First read the existing test file to understand the existing pattern:

```bash
cat tests/unit/servers/web/test_web_experience_app.py
```

Then add the following test at the bottom of `tests/unit/servers/web/test_web_experience_app.py`:

```python
def test_direct_search_enriches_web_provider_content(monkeypatch):
    """Content fetching is called for serpapi/google providers, not for retrieval."""
    from src.tools.search import SearchPage
    from src.backend.servers.web.app import _run_direct_search
    import asyncio

    serpapi_pages = [
        SearchPage(title="Result A", summary="snippet A", url="https://a.test"),
    ]
    fetched_pages = [
        SearchPage(title="Result A", summary="full article content A", url="https://a.test"),
    ]

    async def _fake_search_tool(query, *, provider, search_url, page_size):
        return serpapi_pages

    async def _fake_fetch_pages(pages, *, max_chars, timeout_seconds=10):
        assert pages == serpapi_pages
        return fetched_pages

    monkeypatch.setattr("src.backend.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr("src.backend.servers.web.app.fetch_pages_concurrently", _fake_fetch_pages)

    docs = asyncio.run(
        _run_direct_search(
            "test query",
            source_provider="serpapi",
            search_url="http://localhost:8000/retrieve",
            top_k=3,
        )
    )
    assert any("full article content A" in doc.content for doc in docs)


def test_direct_search_skips_fetch_for_retrieval_provider(monkeypatch):
    """Content fetching is NOT called for the local retrieval provider."""
    from src.tools.search import SearchPage
    from src.backend.servers.web.app import _run_direct_search
    import asyncio

    fetch_called = []

    async def _fake_search_tool(query, *, provider, search_url, page_size):
        return [SearchPage(title="R", summary="corpus content", url="https://r.test")]

    async def _fake_fetch_pages(pages, **kwargs):
        fetch_called.append(True)
        return pages

    monkeypatch.setattr("src.backend.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr("src.backend.servers.web.app.fetch_pages_concurrently", _fake_fetch_pages)

    asyncio.run(
        _run_direct_search(
            "test query",
            source_provider="retrieval",
            search_url="http://localhost:8000/retrieve",
            top_k=3,
        )
    )
    assert not fetch_called
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/servers/web/test_web_experience_app.py::test_direct_search_enriches_web_provider_content \
       tests/unit/servers/web/test_web_experience_app.py::test_direct_search_skips_fetch_for_retrieval_provider \
       -v
```

Expected: `FAILED` — `fetch_pages_concurrently` not imported and not wired.

- [ ] **Step 3: Implement the changes in `src/backend/servers/web/app.py`**

Add the import at the top of the file alongside the existing `search_tool` import (around line 77):

```python
from src.tools.search import SearchPage
from src.tools.search import fetch_pages_concurrently
from src.tools.search import search_tool
```

Add a helper function near the other `_run_*` helpers (after `_source_providers_for`, around line 619):

```python
_WEB_PROVIDERS = {"serpapi", "google"}


def _is_web_provider(source_provider: str) -> bool:
    """Returns True for providers that return URL snippets needing full-page fetch."""
    return source_provider in _WEB_PROVIDERS
```

Replace `_run_direct_search` (lines 633–660):

```python
async def _run_direct_search(
    query: str,
    *,
    source_provider: str,
    search_url: str,
    top_k: int,
) -> list[ContextDocument]:
    # Over-fetch so MMR has candidates beyond top_k to diversify from.
    fetch_k = top_k * 2
    documents: list[ContextDocument] = []
    for provider in _source_providers_for(source_provider):
        pages = await search_tool(
            query,
            provider=_tool_provider_for(provider),
            search_url=search_url,
            page_size=fetch_k,
        )
        if _is_web_provider(provider):
            pages = await fetch_pages_concurrently(pages, max_chars=2000)
        documents.extend(
            _documents_from_search_pages(
                pages,
                source_provider=provider,
                query=query,
                start_index=len(documents) + 1,
            )
        )
    deduped = _dedupe_documents(documents)
    diversified = mmr_rerank(deduped, topk=top_k)
    return _reindex_documents(diversified)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/servers/web/test_web_experience_app.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backend/servers/web/app.py tests/unit/servers/web/test_web_experience_app.py
git commit -m "feat: enrich web provider results with full page content in direct search"
```

---

### Task 4: Wire content fetching and temporal context into hybrid search mode

**Files:**
- Modify: `src/backend/servers/web/app.py`
- Test: `tests/unit/servers/web/test_web_experience_app.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/servers/web/test_web_experience_app.py`:

```python
def test_hybrid_search_enriches_serpapi_provider_content(monkeypatch):
    """Hybrid search fetches full page content for serpapi results."""
    from src.tools.search import SearchPage
    from src.backend.servers.web.app import _run_hybrid_search
    from src.context.models import LLMResponse
    import asyncio

    pages = [SearchPage(title="T", summary="snippet", url="https://t.test")]
    fetched = [SearchPage(title="T", summary="full article body", url="https://t.test")]

    async def _fake_search_tool(query, *, provider, search_url, page_size):
        return pages

    async def _fake_fetch_pages(pgs, *, max_chars, timeout_seconds=10):
        return fetched

    monkeypatch.setattr("src.backend.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr("src.backend.servers.web.app.fetch_pages_concurrently", _fake_fetch_pages)
    monkeypatch.setattr(
        "src.backend.servers.web.app.expand_keywords",
        lambda query, llm: [],
    )

    result = asyncio.run(
        _run_hybrid_search(
            "latest AI news",
            llm=None,
            search_url="http://localhost:8000/retrieve",
            top_k=3,
            filters=None,
            source_provider="serpapi",
        )
    )
    assert any("full article body" in doc.content for doc in result.documents)


def test_hybrid_search_includes_temporal_variant_for_time_sensitive_query(monkeypatch):
    """Temporal variant is added to executed queries for time-sensitive queries."""
    from src.backend.servers.web.app import _run_hybrid_search
    from src.tools.search import SearchPage
    import asyncio

    executed: list[str] = []

    async def _fake_search_tool(query, *, provider, search_url, page_size):
        executed.append(query)
        return [SearchPage(title="T", summary="s", url="https://t.test")]

    async def _fake_fetch_pages(pages, **kwargs):
        return pages

    monkeypatch.setattr("src.backend.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr("src.backend.servers.web.app.fetch_pages_concurrently", _fake_fetch_pages)
    monkeypatch.setattr(
        "src.backend.servers.web.app.expand_keywords",
        lambda query, llm: [],
    )

    result = asyncio.run(
        _run_hybrid_search(
            "latest AI models",
            llm=None,
            search_url="http://localhost:8000/retrieve",
            top_k=3,
            filters=None,
            source_provider="serpapi",
        )
    )
    from datetime import datetime
    year = str(datetime.now().year)
    assert any(year in q for q in result.executed_queries)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/servers/web/test_web_experience_app.py::test_hybrid_search_enriches_serpapi_provider_content \
       tests/unit/servers/web/test_web_experience_app.py::test_hybrid_search_includes_temporal_variant_for_time_sensitive_query \
       -v
```

Expected: `FAILED`.

- [ ] **Step 3: Update imports in `src/backend/servers/web/app.py`**

Add alongside the existing `expand_keywords` import (line 25):

```python
from src.backend.secondary_llm_flows import expand_keywords
from src.backend.secondary_llm_flows.query_expansion import with_temporal_context
```

- [ ] **Step 4: Update `_expanded_queries` and `_run_hybrid_search` in `src/backend/servers/web/app.py`**

Replace `_expanded_queries` (lines 728–736):

```python
def _expanded_queries(query: str, llm: LLMClient | None) -> list[str]:
    if llm is None:
        expansions = []
    else:
        try:
            expansions = expand_keywords(query, llm)
        except Exception:
            logger.exception("Query expansion failed for hybrid web search")
            expansions = []
    queries = [query] + [e for e in expansions if e != query]
    temporal = with_temporal_context(query)
    if temporal != query and temporal not in queries:
        queries.append(temporal)
    return queries
```

Replace the serpapi/google path in `_run_hybrid_search` (lines 701–725) to add content fetching:

```python
    executed_queries = _expanded_queries(query, llm)
    documents: list[ContextDocument] = []
    for provider in _source_providers_for(source_provider):
        for expanded_query in executed_queries:
            pages = await search_tool(
                expanded_query,
                provider=_tool_provider_for(provider),
                search_url=search_url,
                page_size=top_k,
            )
            if _is_web_provider(provider):
                pages = await fetch_pages_concurrently(pages, max_chars=2000)
            documents.extend(
                _documents_from_search_pages(
                    pages,
                    source_provider=provider,
                    query=expanded_query,
                    start_index=len(documents) + 1,
                    entry_point="hybrid_search",
                )
            )
    deduped = _dedupe_documents(documents)
    diversified = mmr_rerank(deduped, topk=top_k)
    return _HybridSearchResult(
        executed_queries=executed_queries,
        documents=_reindex_documents(diversified),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/servers/web/test_web_experience_app.py -v
```

Expected: All tests PASS.

- [ ] **Step 6: Run the full unit suite**

```bash
pytest tests/unit/ -v
```

Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/backend/servers/web/app.py tests/unit/servers/web/test_web_experience_app.py
git commit -m "feat: enrich hybrid search results with page content and inject temporal context"
```

---

## Self-review

### Spec coverage
- [x] Full page content fetching for web providers (Tasks 1, 3, 4)
- [x] Better HTML extraction preferring article/main content (Task 1)
- [x] Concurrent fetching to avoid latency penalty (Task 1 — `asyncio.gather`)
- [x] Temporal query injection for time-sensitive queries (Tasks 2, 4)
- [x] Works for both direct search and hybrid search modes (Tasks 3, 4)
- [x] Local retrieval provider skipped (Tasks 3, 4 — `_is_web_provider` guard)

### No placeholder scan
- All code is complete and concrete. No TBD sections.

### Type consistency
- `fetch_pages_concurrently(pages: list[SearchPage], *, max_chars: int, timeout_seconds: int) -> list[SearchPage]` — used consistently in Tasks 1, 3, 4.
- `_is_web_provider(source_provider: str) -> bool` — defined in Task 3, reused in Task 4.
- `with_temporal_context(query: str) -> str` — defined in Task 2, imported and used in Task 4.
- `is_time_sensitive(query: str) -> bool` — defined in Task 2, tested separately, used inside `with_temporal_context`.
