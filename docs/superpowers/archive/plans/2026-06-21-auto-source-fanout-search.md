# Auto Source Fan-out Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On a `search` intent, always fan out to internal RAG + SerpAPI in parallel, merge into one ranked list, degrade silently when a provider is unavailable, and remove the Source picker from the normal UI (kept behind `?dev=1`).

**Architecture:** Introduce an `auto` source provider that expands to `["retrieval", "serpapi"]`. The auto-router uses it as the default. `_run_hybrid_search`'s multi-provider path is refactored to run providers concurrently with a per-provider guard that converts failures/timeouts into an error-marked document; a finalize step filters error docs, then reports a status (`ok` / `empty` / `unreachable`) that the auto-router turns into the user-facing message. The frontend drops the dropdown except under `?dev=1` and omits `source_provider` in normal mode.

**Tech Stack:** Python 3.12, FastAPI, asyncio; React 19 + TypeScript + Vitest.

## Global Constraints

- Never commit to `main`; this work is on branch `feat/auto-source-fanout-search`. Run `git branch --show-current` before every commit.
- Browser provider (~5–10s/query) stays out of the default fan-out; only `retrieval` + `serpapi`.
- `chat` and `tool` intents must remain unchanged (regression guard required).
- Backend lint: `ruff check . --fix && ruff format .` must pass. Frontend: `npm run typecheck` clean.
- Test commands: backend `PYTHONPATH=src:. python -m pytest <path> -q`; frontend `cd web && npx vitest run <path>`.

---

### Task 1: `auto` provider + default fan-out set

**Files:**
- Modify: `src/internal/servers/web/app.py` (constants ~1242-1271; request model ~163-176; `explicit_source` line 309)
- Test: `tests/unit/test_execution_fallbacks.py`

**Interfaces:**
- Produces: `_DEFAULT_FANOUT_PROVIDERS = ["retrieval", "serpapi"]`; `_source_providers_for("auto") -> ["retrieval", "serpapi"]`; `AgentExperienceRequest.source_provider` default `"auto"`; `explicit_source = source_provider != "auto"`.

- [ ] **Step 1: Write the failing unit test**

In `tests/unit/test_execution_fallbacks.py`, append:

```python
def test_auto_provider_expands_to_internal_and_serpapi():
    from src.internal.servers.web.app import _source_providers_for

    assert _source_providers_for("auto") == ["retrieval", "serpapi"]


def test_auto_is_default_and_not_treated_as_explicit(monkeypatch, tmp_path):
    """No source_provider → 'auto' → classifier still runs (not forced)."""
    from src.internal.servers.web.app import _HybridSearchResult

    captured = {}

    async def fake_hybrid(query, **kwargs):
        captured["source_provider"] = kwargs.get("source_provider")
        doc = ContextDocument(id="D1", title="t", content="c", url=None, score=0.0)
        return _HybridSearchResult(executed_queries=[query], documents=[doc])

    monkeypatch.setattr("src.internal.servers.web.app._run_hybrid_search", fake_hybrid)
    monkeypatch.setattr(
        "src.internal.servers.web.app._rule_based_is_search", lambda q: True
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "anything searchable"})
    assert response.status_code == 200
    assert response.json()["intent"] == "search"
    assert captured["source_provider"] == "auto"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:. python -m pytest tests/unit/test_execution_fallbacks.py::test_auto_provider_expands_to_internal_and_serpapi tests/unit/test_execution_fallbacks.py::test_auto_is_default_and_not_treated_as_explicit -q`
Expected: FAIL — `_source_providers_for("auto")` returns `["auto"]`; second test captures `"retrieval"` (old default) or 422.

- [ ] **Step 3: Add the `auto` provider constants**

Replace the constants block at `src/internal/servers/web/app.py:1242-1271`:

```python
_VALID_SOURCE_PROVIDERS = {
    "auto",
    "retrieval",
    "serpapi",
    "browser",
    "all",
}
_SOURCE_PROVIDER_LABELS = {
    "auto": "Auto (internal + web)",
    "retrieval": "Local Retrieval",
    "serpapi": "SerpAPI",
    "browser": "Browser Retrieval",
    "all": "All Active Sources",
}


def _normalize_source_provider(source_provider: str) -> str:
    requested = source_provider.strip().lower()
    normalized = _SOURCE_PROVIDER_ALIASES.get(requested, requested)
    if normalized not in _VALID_SOURCE_PROVIDERS:
        valid = ", ".join(sorted(_VALID_SOURCE_PROVIDERS))
        raise HTTPException(
            status_code=422,
            detail=f"source_provider must be one of: {valid}",
        )
    return normalized


# Default provider set when the user does not pick a source: fan out to internal
# RAG + fast web search, merged. Browser is excluded (too slow for the default).
_DEFAULT_FANOUT_PROVIDERS = ["retrieval", "serpapi"]


def _source_providers_for(source_provider: str) -> list[str]:
    if source_provider == "all":
        return ["retrieval", "serpapi", "browser"]
    if source_provider == "auto":
        return list(_DEFAULT_FANOUT_PROVIDERS)
    return [source_provider]
```

(The `_SOURCE_PROVIDER_ALIASES` dict directly above line 1242 is unchanged.)

- [ ] **Step 4: Change the request default to `auto`**

At `src/internal/servers/web/app.py:163`, change the field default and description:

```python
    source_provider: str = Field(
        default="auto",
        description=(
            "'auto' (default — fan out to internal RAG + SerpAPI, merged), "
            "'retrieval', 'serpapi', 'browser', or 'all'. An explicit value other "
            "than 'auto' forces a search against that single provider (dev only)."
        ),
    )
```

- [ ] **Step 5: Update `explicit_source` to key off `auto`**

At `src/internal/servers/web/app.py:309`, change:

```python
    explicit_source = source_provider != "auto"
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. python -m pytest tests/unit/test_execution_fallbacks.py -q`
Expected: PASS (including the two new tests and the pre-existing `test_default_source_still_auto_routes_to_chat`, `test_explicit_source_forces_search_against_that_provider`).

- [ ] **Step 7: Commit**

```bash
git add src/internal/servers/web/app.py tests/unit/test_execution_fallbacks.py
git commit -m "feat(search): add 'auto' provider as default fan-out (internal + serpapi)"
```

---

### Task 2: Concurrent fan-out with graceful degradation + per-provider timeout

**Files:**
- Modify: `src/internal/servers/web/app.py` (`_HybridSearchResult` ~217-220; `_documents_from_search_pages` ~1574; `_run_hybrid_search` multi-provider block ~1486-1553; single-provider returns ~1454, 1481)
- Test: `tests/unit/servers/web/test_web_experience_app.py`

**Interfaces:**
- Consumes: `_source_providers_for` (Task 1), `search_tool`, `_documents_from_search_pages`, `_dedupe_documents`, `_rerank_documents`, `mmr_rerank`, `_reindex_documents`, `_run_browser_search`, `fetch_pages_concurrently`.
- Produces: `_HybridSearchResult.status: str` (`"ok"|"empty"|"unreachable"`); error docs carry `metadata["error"] = True`; providers run concurrently; `_provider_error_doc(provider, message)`; `_finalize_hybrid(documents, *, executed_queries, query, rerank_url, top_k)`.

- [ ] **Step 1: Write failing tests**

In `tests/unit/servers/web/test_web_experience_app.py`, add (the helper `SearchPage` is imported as `from src.tools import SearchPage` — add that import at the top of the test file if absent):

```python
def test_hybrid_fanout_merges_real_and_drops_errored_provider(monkeypatch, tmp_path):
    """retrieval returns real pages, serpapi errors → only real docs, status ok."""
    import asyncio
    from src.internal.servers.web.app import _run_hybrid_search
    from src.tools import SearchPage

    async def fake_search_tool(query, *, provider, search_url, page_size, **kw):
        if provider == "retrieval":
            return [SearchPage(title="Real Doc", summary="real", url="http://x/1")]
        return [SearchPage(error="SERPAPI_API_KEY is required.")]

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app._expanded_queries", lambda q, llm: [q]
    )
    result = asyncio.run(
        _run_hybrid_search(
            "q",
            llm=None,
            search_url="http://localhost:8001/retrieve",
            top_k=3,
            filters=None,
            source_provider="auto",
        )
    )
    assert result.status == "ok"
    assert [d.title for d in result.documents] == ["Real Doc"]
    assert all(not d.metadata.get("error") for d in result.documents)


def test_hybrid_fanout_all_errored_is_unreachable(monkeypatch, tmp_path):
    import asyncio
    from src.internal.servers.web.app import _run_hybrid_search
    from src.tools import SearchPage

    async def fake_search_tool(query, *, provider, search_url, page_size, **kw):
        return [SearchPage(error="down")]

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app._expanded_queries", lambda q, llm: [q]
    )
    result = asyncio.run(
        _run_hybrid_search(
            "q", llm=None, search_url="http://x/retrieve", top_k=3,
            filters=None, source_provider="auto",
        )
    )
    assert result.status == "unreachable"
    assert result.documents == []


def test_hybrid_fanout_reachable_but_empty_is_empty(monkeypatch, tmp_path):
    import asyncio
    from src.internal.servers.web.app import _run_hybrid_search

    async def fake_search_tool(query, *, provider, search_url, page_size, **kw):
        return []  # reachable, no hits, no error

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app._expanded_queries", lambda q, llm: [q]
    )
    result = asyncio.run(
        _run_hybrid_search(
            "q", llm=None, search_url="http://x/retrieve", top_k=3,
            filters=None, source_provider="auto",
        )
    )
    assert result.status == "empty"
    assert result.documents == []


def test_hybrid_fanout_one_provider_raises_does_not_kill_other(monkeypatch, tmp_path):
    import asyncio
    from src.internal.servers.web.app import _run_hybrid_search
    from src.tools import SearchPage

    async def fake_search_tool(query, *, provider, search_url, page_size, **kw):
        if provider == "serpapi":
            raise RuntimeError("boom")
        return [SearchPage(title="Real", summary="r", url="http://x/1")]

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app._expanded_queries", lambda q, llm: [q]
    )
    result = asyncio.run(
        _run_hybrid_search(
            "q", llm=None, search_url="http://x/retrieve", top_k=3,
            filters=None, source_provider="auto",
        )
    )
    assert result.status == "ok"
    assert [d.title for d in result.documents] == ["Real"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:. python -m pytest tests/unit/servers/web/test_web_experience_app.py -k hybrid_fanout -q`
Expected: FAIL — `_HybridSearchResult` has no `status`; error docs not filtered.

- [ ] **Step 3: Add `status` to `_HybridSearchResult`**

At `src/internal/servers/web/app.py:217-220`:

```python
@dataclass(frozen=True)
class _HybridSearchResult:
    executed_queries: list[str]
    documents: list[ContextDocument]
    status: str = "ok"  # "ok" | "empty" | "unreachable"
```

- [ ] **Step 4: Mark error documents in `_documents_from_search_pages`**

At `src/internal/servers/web/app.py:1574-1579`, add the `error` key to metadata:

```python
                metadata={
                    "entry_point": entry_point,
                    "source": _source_label(source_provider),
                    "source_provider": source_provider,
                    "query": query,
                    "error": bool(page.error),
                },
```

- [ ] **Step 5: Add `_provider_error_doc` and `_finalize_hybrid` helpers**

Insert directly above `def _run_hybrid_search(` (around line 1426):

```python
def _provider_error_doc(provider: str, message: str) -> ContextDocument:
    """A placeholder doc marking a provider that failed/timed out. Filtered from
    results; its presence signals 'unreachable' when no real docs were found."""
    return ContextDocument(
        id="D0",
        title="Search error",
        content=message,
        url=None,
        score=0.0,
        metadata={
            "source_provider": provider,
            "source": _source_label(provider),
            "error": True,
        },
    )


async def _finalize_hybrid(
    documents: list[ContextDocument],
    *,
    executed_queries: list[str],
    query: str,
    rerank_url: str | None,
    top_k: int,
) -> _HybridSearchResult:
    real = [d for d in documents if not d.metadata.get("error")]
    errored = [d for d in documents if d.metadata.get("error")]
    if not real:
        status = "unreachable" if errored else "empty"
        return _HybridSearchResult(
            executed_queries=executed_queries, documents=[], status=status
        )
    deduped = _dedupe_documents(real)
    if rerank_url:
        deduped = await _rerank_documents(deduped, query, rerank_url)
    diversified = mmr_rerank(deduped, topk=top_k)
    return _HybridSearchResult(
        executed_queries=executed_queries,
        documents=_reindex_documents(diversified),
        status="ok",
    )
```

- [ ] **Step 6: Refactor the multi-provider block to run concurrently with a guard**

Replace the block at `src/internal/servers/web/app.py:1486-1553` (from `executed_queries = _expanded_queries(...)` through the final `return _HybridSearchResult(...)` of the multi-provider path) with:

```python
    executed_queries = _expanded_queries(query, llm)

    async def _fetch_provider(provider: str) -> list[ContextDocument]:
        if provider == "browser":
            if not browser_search_url:
                return []
            return await _run_browser_search(
                query,
                browser_search_url=browser_search_url,
                top_k=top_k * 2,
                existing_count=0,
            )
        page_lists: list[list[SearchPage]] = list(
            await asyncio.gather(
                *[
                    search_tool(
                        expanded_query,
                        provider=provider,
                        search_url=search_url,
                        page_size=top_k,
                        timeout_seconds=5,
                        max_retries=1,
                    )
                    for expanded_query in executed_queries
                ]
            )
        )
        if _is_web_provider(provider):
            all_pages = [p for pages in page_lists for p in pages]
            enriched = await fetch_pages_concurrently(all_pages, max_chars=2000)
            it = iter(enriched)
            page_lists = [list(islice(it, len(pages))) for pages in page_lists]
        docs: list[ContextDocument] = []
        for expanded_query, pages in zip(executed_queries, page_lists):
            docs.extend(
                _documents_from_search_pages(
                    pages,
                    source_provider=provider,
                    query=expanded_query,
                    start_index=len(docs) + 1,
                )
            )
        return docs

    async def _fetch_provider_guarded(provider: str) -> list[ContextDocument]:
        try:
            return await asyncio.wait_for(_fetch_provider(provider), timeout=8.0)
        except Exception as exc:  # timeout or provider error → mark unreachable
            logger.warning("Provider %s failed/timed out: %s", provider, exc)
            return [_provider_error_doc(provider, str(exc))]

    provider_docs = await asyncio.gather(
        *[
            _fetch_provider_guarded(provider)
            for provider in _source_providers_for(source_provider)
        ]
    )
    documents = [doc for docs in provider_docs for doc in docs]
    return await _finalize_hybrid(
        documents,
        executed_queries=executed_queries,
        query=query,
        rerank_url=rerank_url,
        top_k=top_k,
    )
```

- [ ] **Step 7: Set status on the single-provider return paths**

At the Path A retrieval return (`src/internal/servers/web/app.py:1454-1465`), change the return to set status:

```python
        return _HybridSearchResult(
            executed_queries=search_result.executed_queries,
            documents=[
                _document_with_metadata(
                    doc,
                    source_provider=source_provider,
                    query=query,
                    entry_point="hybrid_search",
                )
                for doc in diversified
            ],
            status="ok" if diversified else "empty",
        )
```

At the Path B browser return (`src/internal/servers/web/app.py:1481-1484`):

```python
        return _HybridSearchResult(
            executed_queries=[query],
            documents=_reindex_documents(diversified_b),
            status="ok" if diversified_b else "empty",
        )
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. python -m pytest tests/unit/servers/web/test_web_experience_app.py -k hybrid_fanout -q`
Expected: PASS (4 new tests).

- [ ] **Step 9: Run the broader backend suite for regressions**

Run: `PYTHONPATH=src:. python -m pytest tests/unit/servers/web/ tests/unit/test_execution_fallbacks.py -q`
Expected: PASS. If `test_browser_pipeline.py` asserts on sequential ordering or exact doc indices, adjust those assertions to match the merged/reindexed output (documents are reindexed `D1..Dn` after MMR).

- [ ] **Step 10: Lint + commit**

```bash
ruff check src/internal/servers/web/app.py --fix && ruff format src/internal/servers/web/app.py tests/unit/servers/web/test_web_experience_app.py
git add src/internal/servers/web/app.py tests/unit/servers/web/test_web_experience_app.py
git commit -m "feat(search): concurrent provider fan-out with graceful degradation + status"
```

---

### Task 3: Auto-router surfaces status as user message

**Files:**
- Modify: `src/internal/servers/web/app.py` (`_run_auto_routed` search branch ~435-465)
- Test: `tests/unit/test_execution_fallbacks.py`

**Interfaces:**
- Consumes: `_HybridSearchResult.status` (Task 2), `_search_only_answer`.
- Produces: search-intent answer text that distinguishes `unreachable` ("No sources are reachable right now…") from `empty`/`ok`.

- [ ] **Step 1: Write failing tests**

In `tests/unit/test_execution_fallbacks.py`, append:

```python
def test_search_unreachable_returns_clear_message(monkeypatch, tmp_path):
    from src.internal.servers.web.app import _HybridSearchResult

    async def fake_hybrid(query, **kwargs):
        return _HybridSearchResult(
            executed_queries=[query], documents=[], status="unreachable"
        )

    monkeypatch.setattr("src.internal.servers.web.app._run_hybrid_search", fake_hybrid)
    monkeypatch.setattr(
        "src.internal.servers.web.app._rule_based_is_search", lambda q: True
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "find stuff"})
    data = response.json()
    assert data["intent"] == "search"
    assert "No sources are reachable" in data["answer"]
    assert data["documents"] == []


def test_search_empty_uses_no_results_message(monkeypatch, tmp_path):
    from src.internal.servers.web.app import _HybridSearchResult

    async def fake_hybrid(query, **kwargs):
        return _HybridSearchResult(
            executed_queries=[query], documents=[], status="empty"
        )

    monkeypatch.setattr("src.internal.servers.web.app._run_hybrid_search", fake_hybrid)
    monkeypatch.setattr(
        "src.internal.servers.web.app._rule_based_is_search", lambda q: True
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "find stuff"})
    data = response.json()
    assert data["intent"] == "search"
    assert "No sources are reachable" not in data["answer"]
    assert "no results" in data["answer"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:. python -m pytest tests/unit/test_execution_fallbacks.py -k "unreachable or empty_uses" -q`
Expected: FAIL — current code always uses `_search_only_answer`; no "No sources are reachable" branch.

- [ ] **Step 3: Branch the search answer on status**

Replace `src/internal/servers/web/app.py:435-465` (the `if is_search:` block) with:

```python
    if is_search:
        provider = source_provider
        try:
            search_result = await _run_hybrid_search(
                query,
                llm=llm,
                search_url=search_url,
                browser_search_url=browser_search_url,
                rerank_url=rerank_url,
                top_k=top_k,
                filters=filters,
                source_provider=provider,
            )
        except Exception as exc:
            logger.warning(
                "Hybrid search failed, falling back to RAG without context: %s", exc
            )
            extra["search_fallback"] = "retrieval_unavailable"
        else:
            if search_result.status == "unreachable":
                query_lines = "\n".join(
                    f"- {q}" for q in search_result.executed_queries
                )
                answer = (
                    "No sources are reachable right now. Please try again shortly.\n\n"
                    f"Executed queries:\n{query_lines}"
                )
            else:
                answer = _search_only_answer(
                    "Search",
                    queries=search_result.executed_queries,
                    documents=search_result.documents,
                    source_provider=provider,
                )
            return (
                answer,
                [d.citation for d in search_result.documents],
                search_result.documents,
                "search",
                extra,
            )
```

(The `# Chat path (also search fallback)` block immediately below is unchanged and now handles the `except` fall-through.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. python -m pytest tests/unit/test_execution_fallbacks.py -q`
Expected: PASS (new tests + existing `test_default_source_still_auto_routes_to_chat` chat regression guard).

- [ ] **Step 5: Lint + commit**

```bash
ruff check src/internal/servers/web/app.py --fix && ruff format src/internal/servers/web/app.py tests/unit/test_execution_fallbacks.py
git add src/internal/servers/web/app.py tests/unit/test_execution_fallbacks.py
git commit -m "feat(search): distinguish unreachable vs empty in search answer"
```

---

### Task 4: Frontend — remove dropdown from normal UI, gate behind `?dev=1`

**Files:**
- Modify: `web/src/types.ts` (`SearchSourceProvider` union ~27), `web/src/components/SearchComposer.tsx`, `web/src/App.tsx`
- Test: `web/src/components/__tests__/SearchComposer.test.tsx`, `web/src/components/__tests__/App.test.tsx`

**Interfaces:**
- Consumes: existing `DEV_MODE` constant in `App.tsx`, `showUrlField` pattern.
- Produces: `SearchComposer` prop `showSourcePicker?: boolean` (default `false`); `SearchSourceProvider` includes `"auto"`; normal-mode request omits `source_provider`.

- [ ] **Step 1: Write failing frontend tests**

In `web/src/components/__tests__/SearchComposer.test.tsx`, add:

```typescript
  it("hides the Source dropdown by default", () => {
    render(<SearchComposer {...defaultProps} />);
    expect(screen.queryByText("Local Retrieval")).not.toBeInTheDocument();
  });

  it("shows the Source dropdown when showSourcePicker is set (dev mode)", () => {
    render(<SearchComposer {...defaultProps} showSourcePicker />);
    expect(screen.getByText("Local Retrieval")).toBeInTheDocument();
  });
```

In `web/src/components/__tests__/App.test.tsx`, update the existing test in `describe("App retrieval URL handling"...)` so it asserts `source_provider` is omitted in normal mode. Change the assertion block in `"does not send a client search_url in normal (non-dev) mode"`:

```typescript
    const sentRequest = mockStreamAgent.mock.calls[0][0] as {
      search_url?: string;
      source_provider?: string;
    };
    expect(sentRequest.search_url).toBeUndefined();
    expect(sentRequest.source_provider).toBeUndefined();
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd web && npx vitest run src/components/__tests__/SearchComposer.test.tsx src/components/__tests__/App.test.tsx`
Expected: FAIL — dropdown renders by default; `source_provider` is `"retrieval"`, not undefined.

- [ ] **Step 3: Add `"auto"` to the provider type**

In `web/src/types.ts`, extend the `SearchSourceProvider` union (line ~27) to include `"auto"` as the first member, e.g.:

```typescript
export type SearchSourceProvider =
  | "auto"
  | "retrieval"
  | "serpapi"
  | "browser"
  | "all";
```

(Keep any existing members; just add `"auto"`. Verify the exact current members before editing.)

- [ ] **Step 4: Gate the Source dropdown behind `showSourcePicker`**

In `web/src/components/SearchComposer.tsx`:

1. Add to `SOURCE_OPTIONS` (top of file) an auto entry as the first item: `{ value: "auto", label: "Auto (internal + web)" },`.
2. Add `showSourcePicker?: boolean;` to `SearchComposerProps` (near `showUrlField`).
3. Add `showSourcePicker = false,` to the destructured params (near `showUrlField = false,`).
4. Wrap the `Source` `<label>…</label>` block (the `<select>`) in `{showSourcePicker && ( … )}`.

- [ ] **Step 5: Wire `App.tsx` — default `auto`, gate picker, omit `source_provider` in normal mode**

In `web/src/App.tsx`:

1. Change the source state default (line ~48-49): `useState<SearchSourceProvider>("auto");`
2. In the request body, change `source_provider: sourceProvider,` to:

```typescript
        source_provider: DEV_MODE ? sourceProvider : undefined,
```

3. In the `<SearchComposer .../>` props, add `showSourcePicker={DEV_MODE}` (next to `showUrlField={DEV_MODE}`).

- [ ] **Step 6: Run typecheck + tests to verify they pass**

Run: `cd web && npm run typecheck && npx vitest run`
Expected: PASS — typecheck clean; all vitest suites green.

- [ ] **Step 7: Commit**

```bash
git add web/src/types.ts web/src/components/SearchComposer.tsx web/src/App.tsx web/src/components/__tests__/SearchComposer.test.tsx web/src/components/__tests__/App.test.tsx
git commit -m "feat(web): drop Source picker from normal UI; gate behind ?dev=1, omit source_provider"
```

---

### Task 5: End-to-end verification, full suites, PR

**Files:** none (verification + docs)

- [ ] **Step 1: Rebuild frontend bundle**

Run: `cd web && npm run build`
Expected: build succeeds; FastAPI will serve the new `web/dist`.

- [ ] **Step 2: Restart stack and verify fan-out + degradation manually**

With retrieval demo on 8001 and the web backend on 7860 (restart it to pick up code), run:

```bash
curl -s -X POST http://localhost:7860/api/agent -H "Content-Type: application/json" \
  -d '{"query":"What is FAISS?","top_k":3}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('intent=',d['intent']); [print(x['citation'],x['metadata'].get('source'),x['title']) for x in d['documents']]"
```

Expected: `intent=search`; documents include `source` of both `Local Retrieval` and `SerpAPI` (if `SERP_API_KEY` set), no `Search error` cards.

- [ ] **Step 3: Run full backend + frontend suites**

Run: `PYTHONPATH=src:. python -m pytest tests/unit -q && cd web && npx vitest run && npm run typecheck`
Expected: all green.

- [ ] **Step 4: Push and open PR**

```bash
git push -u origin feat/auto-source-fanout-search
gh pr create --base main --title "feat(search): auto source fan-out (internal + web), no source picker" --body "<summary referencing spec + plan>"
```

---

## Self-Review

**Spec coverage:**
- Always fan out internal + SerpAPI merged → Task 1 (`auto`=`[retrieval,serpapi]`) + Task 2 (concurrent fan-out + merge). ✅
- Browser opt-in/dev-only → Task 1 (excluded from `auto`), Task 4 (selectable only under `?dev=1`). ✅
- Chat/tool unchanged → Task 2 Step 9 + Task 3 Step 4 regression guards. ✅
- Graceful degradation (SerpAPI missing → internal-only; internal down → SerpAPI-only; both → message; independent providers; timeout) → Task 2 (guard, `_finalize_hybrid`, `wait_for(8s)`, `timeout_seconds=5`) + Task 3 (unreachable message). ✅
- Remove dropdown from normal UI, keep under `?dev=1`, omit `source_provider` → Task 4. ✅
- Per-result provenance preserved → unchanged `source`/`source_provider` metadata (Task 2 keeps it). ✅
- Tests for all of the above → Tasks 1-4 each TDD. ✅

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✅

**Type consistency:** `_HybridSearchResult.status` defined in Task 2 Step 3, consumed in Task 3 Step 3. `_source_providers_for("auto")` defined Task 1, used Task 2. `showSourcePicker` defined and consumed within Task 4. `_finalize_hybrid`/`_provider_error_doc` defined and used within Task 2. ✅
