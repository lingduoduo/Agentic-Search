# Generated Context Pack

# Auto Source Fanout Search

## Sources

- [Specification: 2026-06-21-auto-source-fanout-search-design.md](../specs/2026-06-21-auto-source-fanout-search-design.md)
- [Plan: 2026-06-21-auto-source-fanout-search.md](../plans/2026-06-21-auto-source-fanout-search.md)

## Specification Context

### Goal

On a `search` intent, always fan out to internal RAG **and** SerpAPI in parallel, merge
into one ranked list, and remove the Source picker from the normal UI. Unconfigured or
failing providers degrade silently. The picker survives only as a `?dev=1` affordance.

### Scope

- `src/internal/servers/web/app.py` (auto-router fan-out + degradation)
- `web/src/App.tsx`, `web/src/components/SearchComposer.tsx` (remove dropdown from normal UI)
- Tests (backend fan-out + degradation; frontend dropdown gating)

Out of scope: changing the `chat` / `tool` intents; browser in the default path;
per-provider UI configuration; reranker wiring.

### Testing

**Backend**
- Search intent fans out to internal + SerpAPI and returns a merged/deduped list (mock both
  providers; assert documents from each appear).
- SerpAPI unconfigured → internal-only, 200, no error.
- Internal down → SerpAPI-only.
- Both down → clear "no sources reachable" message, not a silent empty.
- One provider raising/timing out does not fail the other.
- `chat` intent still grounds internal-only (regression guard).

**Frontend**
- Source dropdown absent in normal mode; present under `?dev=1`.
- Request omits `source_provider` in normal mode.

## Implementation Plan Context

### Global Constraints

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

_[Section compacted.]_

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

_[Section compacted.]_

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

_[Section compacted.]_

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

_[Section compacted.]_

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

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
