# Browser Hybrid Search + Rerank + MMR Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `browser.py` fetch full page content (not just SERP snippets), merge browser results with other search providers, apply cross-encoder reranking before MMR, and display MMR rank + color-coded scores in the UI.

**Architecture:** (1) `browser.py` gains a second phase: after extracting URLs from the SERP, navigate to each URL with playwright-cli and extract full page text; (2) `SearchExperienceSettings` gains two optional URLs — `browser_search_url` (a running browser.py server to merge results from) and `rerank_url` (a running rerank.py server); (3) `_run_direct_search` and `_run_hybrid_search` in `app.py` merge browser docs with the primary provider, call the reranker to assign real relevance scores, then run MMR; (4) `_reindex_documents` stamps each doc with `mmr_rank` so the frontend can display it; (5) `SourceGrid.tsx` renders color-coded score badges and an MMR rank number.

**Tech Stack:** playwright-cli (subprocess), Python `dataclasses.replace`-style construction, `httpx.AsyncClient` (new import in app.py), React 19 + TypeScript inline styles.

---

## File Map

| File | Change |
|------|--------|
| `src/internal/servers/web_search/browser.py` | Add `_PAGE_TEXT_JS`, `fetch_content`/`content_timeout` to config, `_fetch_page_text` method, update `_search_and_process` |
| `tests/unit/retrieval/test_browser_retrieval.py` | Add `fetch_content=False` to all 8 existing configs; add 2 new tests |
| `src/internal/servers/web/app.py` | Add `browser_search_url`/`rerank_url` to `SearchExperienceSettings`; add `import httpx`; add `_run_browser_search` and `_rerank_documents`; extend `_run_direct_search` and `_run_hybrid_search` (all-provider branch); stamp `mmr_rank` in `_reindex_documents` |
| `tests/unit/servers/web/test_reranking.py` | New file: tests for `browser_search_url` merging and `rerank_url` calling |
| `web/src/components/SourceGrid.tsx` | Color-coded score badge, MMR rank number, per-provider source color |
| `web/src/components/__tests__/SourceGrid.test.tsx` | Add 2 new tests for mmr_rank badge and score color |

---

### Task 1: browser.py — Two-phase retrieval (search SERP → fetch page content)

**Files:**
- Modify: `src/internal/servers/web_search/browser.py`
- Modify: `tests/unit/retrieval/test_browser_retrieval.py`

- [ ] **Step 1: Add `fetch_content=False` to all 8 existing test configs**

Open `tests/unit/retrieval/test_browser_retrieval.py`. Every `BrowserSearchConfig(...)` call must gain `fetch_content=False` so existing tests keep passing after the default changes to `True`.

```python
# test_search_query_returns_formatted_documents
engine = BrowserSearchEngine(BrowserSearchConfig(topk=2, fetch_content=False))

# test_empty_results_when_eval_returns_empty_list
engine = BrowserSearchEngine(BrowserSearchConfig(topk=5, fetch_content=False))

# test_subprocess_timeout_returns_empty_and_closes
engine = BrowserSearchEngine(BrowserSearchConfig(topk=5, fetch_content=False))

# test_topk_truncates_results
engine = BrowserSearchEngine(BrowserSearchConfig(topk=1, fetch_content=False))

# test_batch_search_runs_queries_in_parallel
engine = BrowserSearchEngine(BrowserSearchConfig(topk=2, batch_workers=2, fetch_content=False))

# test_non_dict_eval_items_are_filtered
engine = BrowserSearchEngine(BrowserSearchConfig(topk=5, fetch_content=False))

# test_quoted_json_eval_output_is_decoded
engine = BrowserSearchEngine(BrowserSearchConfig(topk=5, fetch_content=False))

# test_google_empty_falls_back_to_wikipedia_article
engine = BrowserSearchEngine(BrowserSearchConfig(topk=5, fetch_content=False))
```

- [ ] **Step 2: Append 2 new failing tests**

Add to the bottom of `tests/unit/retrieval/test_browser_retrieval.py`:

```python
def test_fetch_content_navigates_to_result_url():
    """With fetch_content=True, each hit URL is navigated to and full text replaces snippet."""
    engine = BrowserSearchEngine(BrowserSearchConfig(topk=1, fetch_content=True))
    page_text = "Full article: FAISS is a library for efficient similarity search."
    single_hit = json.dumps([
        {
            "title": "FAISS - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/FAISS",
            "snippet": "A library for similarity search.",
        }
    ])
    with patch("src.internal.servers.web_search.browser.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _make_proc(),                         # open about:blank
            _make_proc(),                         # goto Google search URL
            _make_proc(),                         # snapshot Google
            _make_proc(single_hit),               # eval Google extract JS  → 1 hit, break
            _make_proc(),                         # goto result[0].url
            _make_proc(),                         # snapshot result page
            _make_proc(json.dumps(page_text)),    # eval _PAGE_TEXT_JS
            _make_proc(),                         # close
        ]
        results = engine._search_and_process("what is FAISS")
    assert len(results) == 1
    assert page_text in results[0]["document"]["contents"]
    assert results[0]["document"]["url"] == "https://en.wikipedia.org/wiki/FAISS"


def test_fetch_content_falls_back_to_snippet_on_timeout():
    """When page navigation times out, the original SERP snippet is used instead."""
    engine = BrowserSearchEngine(BrowserSearchConfig(topk=1, fetch_content=True))
    single_hit = json.dumps([
        {
            "title": "FAISS - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/FAISS",
            "snippet": "A library for similarity search.",
        }
    ])
    with patch("src.internal.servers.web_search.browser.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _make_proc(),                                           # open
            _make_proc(),                                           # goto Google
            _make_proc(),                                           # snapshot Google
            _make_proc(single_hit),                                 # eval Google → 1 hit
            subprocess.TimeoutExpired("playwright-cli", 10),        # goto result URL → timeout
            _make_proc(),                                           # close
        ]
        results = engine._search_and_process("what is FAISS")
    assert len(results) == 1
    assert "A library for similarity search." in results[0]["document"]["contents"]
```

- [ ] **Step 3: Run new tests — verify they fail**

```bash
pytest tests/unit/retrieval/test_browser_retrieval.py::test_fetch_content_navigates_to_result_url \
       tests/unit/retrieval/test_browser_retrieval.py::test_fetch_content_falls_back_to_snippet_on_timeout \
       -v
```

Expected: `FAILED` — `BrowserSearchConfig` has no `fetch_content` parameter yet.
Also run all 8 existing tests to confirm they still pass after adding `fetch_content=False`.

- [ ] **Step 4: Add `_PAGE_TEXT_JS`, update `BrowserSearchConfig`, add `_fetch_page_text`**

In `src/internal/servers/web_search/browser.py`:

a) After `_WIKIPEDIA_EXTRACT_JS` (line ~68), add:

```python
_PAGE_TEXT_JS = (
    "JSON.stringify("
    "[...document.querySelectorAll('p,h1,h2,h3,li')]"
    ".map(e=>e.textContent.trim())"
    ".filter(t=>t.length>20)"
    ".slice(0,60)"
    ".join(' ')"
    ")"
)
```

b) Replace `BrowserSearchConfig` (currently lines 77-80):

```python
@dataclass(frozen=True)
class BrowserSearchConfig:
    topk: int = DEFAULT_TOPK
    batch_workers: int = 4
    subprocess_timeout: int = SUBPROCESS_TIMEOUT
    fetch_content: bool = True
    content_timeout: int = 10
```

c) Add `_fetch_page_text` to `BrowserSearchEngine`, after `_extract_hits` (around line 122):

```python
def _fetch_page_text(self, url: str, *, session: str) -> str:
    try:
        self._run("goto", url, session=session)
        self._run("snapshot", session=session)
        proc = self._run("eval", _PAGE_TEXT_JS, session=session, raw=True)
        if proc.stdout.strip():
            raw = json.loads(proc.stdout.strip())
            return str(raw)[:3000] if raw else ""
    except Exception as exc:
        logger.info("fetch_page_text failed for %s: %s", url, exc)
    return ""
```

- [ ] **Step 5: Replace the `return` at end of `_search_and_process`**

Find the current lines 163-166 in `_search_and_process`:

```python
        return [
            format_document(h.get("title"), h.get("snippet"), h.get("url"))
            for h in hits[: self.config.topk]
        ]
```

Replace with:

```python
        results = []
        for h in hits[: self.config.topk]:
            url = h.get("url", "")
            content = h.get("snippet", "")
            if self.config.fetch_content and url:
                fetched = self._fetch_page_text(url, session=session)
                if fetched:
                    content = fetched
            results.append(format_document(h.get("title"), content, url or None))
        return results
```

- [ ] **Step 6: Run all browser tests**

```bash
pytest tests/unit/retrieval/test_browser_retrieval.py -v
```

Expected: 10 passed.

- [ ] **Step 7: Commit**

```bash
git checkout -b feat/browser-hybrid-rerank-mmr
git add src/internal/servers/web_search/browser.py \
        tests/unit/retrieval/test_browser_retrieval.py
git commit -m "feat(browser): two-phase retrieval — search SERP then fetch full page content"
```

---

### Task 2: Hybrid search — merge browser results with primary provider + reranker wiring

**Files:**
- Modify: `src/internal/servers/web/app.py`
- Create: `tests/unit/servers/web/test_reranking.py`

Add `browser_search_url` and `rerank_url` to `SearchExperienceSettings`, thread them through `run_agent`, and add helpers `_run_browser_search` and `_rerank_documents`.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/servers/web/test_reranking.py`:

```python
"""Tests for browser hybrid merging and cross-encoder rerank integration."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.agents.base import AnswerGenerationResult
from src.context.models import ContextDocument, SearchContextBundle
from src.context.prompting import PromptBundle
from src.tools.api import SearchPage
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


def _make_page(title: str, url: str, provider: str) -> SearchPage:
    return SearchPage(title=title, summary=f"summary from {provider}", url=url, score=0.5)


def _fake_answer(question: str) -> AnswerGenerationResult:
    doc = ContextDocument(
        id="D1", title="T", content="c", url="https://t.test", score=0.5
    )
    return AnswerGenerationResult(
        answer="ok",
        citations=["D1"],
        context=SearchContextBundle(query=question, documents=[doc]),
        prompt=PromptBundle(system="", user="", messages=[]),
    )


def test_browser_search_url_causes_browser_provider_call(tmp_path, monkeypatch):
    """When browser_search_url is set, search_tool is called with that URL."""
    call_log: list[tuple[str, str]] = []

    async def fake_search(query, *, provider, search_url, page_size=5):
        call_log.append((provider, search_url))
        return [_make_page("T", "https://t.test", provider)]

    async def fake_answer(question, *, context, llm_client=None):
        return _fake_answer(question)

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search)
    monkeypatch.setattr("src.internal.servers.web.app.answer_with_retrieval", fake_answer)

    app = create_web_app(
        SearchExperienceSettings(
            db_path=tmp_path / "s.sqlite3",
            browser_search_url="http://browser.test:8002",
        )
    )
    client = TestClient(app)
    resp = client.post(
        "/api/agent",
        json={"query": "hybrid test", "mode": "search_tool", "source_provider": "retrieval"},
    )
    assert resp.status_code == 200
    browser_calls = [(p, u) for p, u in call_log if u == "http://browser.test:8002"]
    assert browser_calls, f"browser_search_url not used; calls were {call_log}"


def test_rerank_url_causes_rerank_documents_call(tmp_path, monkeypatch):
    """When rerank_url is set, _rerank_documents is called with that URL."""
    rerank_calls: list[str] = []

    async def fake_search(query, *, provider, search_url, page_size=5):
        return [_make_page("T", "https://t.test", provider)]

    async def fake_answer(question, *, context, llm_client=None):
        return _fake_answer(question)

    async def fake_rerank(docs, query, rerank_url):
        rerank_calls.append(rerank_url)
        return docs

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search)
    monkeypatch.setattr("src.internal.servers.web.app.answer_with_retrieval", fake_answer)
    monkeypatch.setattr("src.internal.servers.web.app._rerank_documents", fake_rerank)

    app = create_web_app(
        SearchExperienceSettings(
            db_path=tmp_path / "s.sqlite3",
            rerank_url="http://rerank.test:6980",
        )
    )
    client = TestClient(app)
    resp = client.post(
        "/api/agent",
        json={"query": "rerank test", "mode": "search_tool"},
    )
    assert resp.status_code == 200
    assert rerank_calls == ["http://rerank.test:6980"]
```

- [ ] **Step 2: Run — verify they fail**

```bash
pytest tests/unit/servers/web/test_reranking.py -v
```

Expected: `FAILED` — `SearchExperienceSettings` has no `browser_search_url` or `rerank_url`.

- [ ] **Step 3: Add new fields to `SearchExperienceSettings`**

In `src/internal/servers/web/app.py`, find `SearchExperienceSettings` (line ~96) and add two new fields with defaults:

```python
@dataclass(frozen=True)
class SearchExperienceSettings:
    """Runtime settings for the browser search app."""

    search_url: str = "http://localhost:8000/retrieve"
    top_k: int = 5
    db_path: str | Path = ":memory:"
    browser_search_url: str | None = None
    rerank_url: str | None = None
```

- [ ] **Step 4: Add `import httpx` to app.py**

At the top of `src/internal/servers/web/app.py`, in the stdlib imports section (around line 5):

```python
import asyncio
import httpx
import json as _json
import logging
```

- [ ] **Step 5: Add `_run_browser_search` helper**

Add this function after `_run_direct_search` (around line 930):

```python
async def _run_browser_search(
    query: str,
    *,
    browser_search_url: str,
    top_k: int,
    existing_count: int,
) -> list[ContextDocument]:
    """Call a running browser.py /retrieve server and convert results to ContextDocuments."""
    try:
        pages = await search_tool(
            query,
            provider="retrieval",
            search_url=browser_search_url,
            page_size=top_k,
        )
    except Exception as exc:
        logger.warning("Browser search failed for %r: %s", query, exc)
        return []
    return _documents_from_search_pages(
        pages,
        source_provider="browser",
        query=query,
        start_index=existing_count + 1,
    )
```

- [ ] **Step 6: Add `_rerank_documents` helper**

Add this function after `_run_browser_search`:

```python
async def _rerank_documents(
    docs: list[ContextDocument],
    query: str,
    rerank_url: str,
) -> list[ContextDocument]:
    """Send docs to the cross-encoder rerank server; update scores and return in ranked order."""
    if not docs:
        return docs
    doc_payloads = [
        {"document": {"contents": f"{d.title}\n{d.content}", "_idx": str(i)}}
        for i, d in enumerate(docs)
    ]
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{rerank_url.rstrip('/')}/rerank",
                json={
                    "queries": [query],
                    "documents": [doc_payloads],
                    "return_scores": True,
                },
                timeout=10.0,
            )
            resp.raise_for_status()
        ranked = resp.json()["result"][0]
        reranked: list[ContextDocument] = []
        for item in ranked:
            idx = int(item["document"].get("_idx", -1))
            score = float(item.get("score", 0.0))
            if 0 <= idx < len(docs):
                orig = docs[idx]
                reranked.append(
                    ContextDocument(
                        id=orig.id,
                        title=orig.title,
                        content=orig.content,
                        url=orig.url,
                        score=score,
                        metadata=orig.metadata,
                    )
                )
        if reranked:
            return reranked
    except Exception as exc:
        logger.warning("Rerank request failed, using original order: %s", exc)
    return docs
```

- [ ] **Step 7: Update `_run_direct_search` signature and body**

Find `_run_direct_search` (around line 900). Change its signature and add browser + rerank calls:

```python
async def _run_direct_search(
    query: str,
    *,
    source_provider: str,
    search_url: str,
    browser_search_url: str | None,
    rerank_url: str | None,
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
    if browser_search_url and source_provider != "browser":
        browser_docs = await _run_browser_search(
            query,
            browser_search_url=browser_search_url,
            top_k=fetch_k,
            existing_count=len(documents),
        )
        documents.extend(browser_docs)
    deduped = _dedupe_documents(documents)
    if rerank_url:
        deduped = await _rerank_documents(deduped, query, rerank_url)
    diversified = mmr_rerank(deduped, topk=top_k)
    return _reindex_documents(diversified)
```

- [ ] **Step 8: Update `_run_hybrid_search` — all-provider branch**

Find `_run_hybrid_search` (around line 932). Add `browser_search_url: str | None` and `rerank_url: str | None` to its signature.

In the `else`/all-provider branch (lines ~969-1008), after the `for provider in _source_providers_for(...)` loop and before `deduped = _dedupe_documents(documents)`, add:

```python
    if browser_search_url and source_provider not in {"browser"}:
        browser_docs = await _run_browser_search(
            query,
            browser_search_url=browser_search_url,
            top_k=top_k * 2,
            existing_count=len(documents),
        )
        documents.extend(browser_docs)
    deduped = _dedupe_documents(documents)
    if rerank_url:
        deduped = await _rerank_documents(deduped, query, rerank_url)
```

- [ ] **Step 9: Thread new params from `run_agent` into both helpers**

In `run_agent` (the `@app.post("/api/agent")` handler), find where `_run_direct_search` is called (line ~458) and add the new keyword args:

```python
documents = await _run_direct_search(
    query,
    source_provider=source_provider,
    search_url=search_url,
    browser_search_url=settings.browser_search_url,
    rerank_url=settings.rerank_url,
    top_k=top_k,
)
```

Find where `_run_hybrid_search` is called (line ~497) and add similarly:

```python
search_result = await _run_hybrid_search(
    query,
    llm=llm,
    search_url=search_url,
    browser_search_url=settings.browser_search_url,
    rerank_url=settings.rerank_url,
    top_k=top_k,
    filters=filters,
    source_provider=source_provider,
)
```

- [ ] **Step 10: Stamp `mmr_rank` in `_reindex_documents`**

Find `_reindex_documents` (around line 1093) and update:

```python
def _reindex_documents(documents: list[ContextDocument]) -> list[ContextDocument]:
    return [
        ContextDocument(
            id=f"D{index}",
            title=document.title,
            content=document.content,
            url=document.url,
            score=document.score,
            metadata={**document.metadata, "mmr_rank": index},
        )
        for index, document in enumerate(documents, 1)
    ]
```

- [ ] **Step 11: Run all tests**

```bash
pytest tests/unit/servers/web/ -v
```

Expected: all pass including the 2 new tests in `test_reranking.py`.

Also run the full suite:
```bash
pytest tests/unit/ -v --tb=short
```

Expected: all pass.

- [ ] **Step 12: Commit**

```bash
git add src/internal/servers/web/app.py \
        tests/unit/servers/web/test_reranking.py
git commit -m "feat(search): hybrid browser merge, cross-encoder rerank before MMR, mmr_rank metadata"
```

---

### Task 3: UI — MMR rank badge and color-coded score in SourceGrid

**Files:**
- Modify: `web/src/components/SourceGrid.tsx`
- Modify: `web/src/components/__tests__/SourceGrid.test.tsx`

- [ ] **Step 1: Add 2 new failing tests**

Open `web/src/components/__tests__/SourceGrid.test.tsx` and add inside the `describe` block:

```tsx
it("shows mmr_rank badge from metadata", () => {
  const ranked = { ...doc, metadata: { source: "Local Retrieval", mmr_rank: 3 } };
  render(<SourceGrid documents={[ranked]} />);
  expect(screen.getByText("#3")).toBeInTheDocument();
});

it("applies green color to high score badge", () => {
  render(<SourceGrid documents={[doc]} />);       // doc.score = 0.95
  const badge = document.querySelector(".score-badge") as HTMLElement;
  expect(badge).not.toBeNull();
  expect(badge.style.color).toBe("rgb(34, 197, 94)");
});
```

Note: the second test uses `document.querySelector` from jsdom (available as a global in the test environment), not `screen`. The `document` global is available in vitest/jsdom.

- [ ] **Step 2: Run — verify they fail**

```bash
cd web && npx vitest run src/components/__tests__/SourceGrid.test.tsx
```

Expected: 2 new tests fail — `.score-badge` class and `#3` badge don't exist yet. The 6 existing tests should still pass.

- [ ] **Step 3: Replace `SourceGrid.tsx`**

Overwrite `web/src/components/SourceGrid.tsx` with:

```tsx
import { memo } from "react";
import { ExternalLink } from "lucide-react";
import type { SourceDocumentView } from "../types";

interface SourceGridProps {
  documents: SourceDocumentView[];
}

function scoreColor(score: number): string {
  if (score >= 0.7) return "rgb(34, 197, 94)";   // green
  if (score >= 0.4) return "rgb(234, 179, 8)";   // yellow
  if (score > 0)   return "rgb(249, 115, 22)";   // orange
  return "rgb(148, 163, 184)";                    // gray (no score)
}

const SOURCE_COLORS: Record<string, string> = {
  "Browser Retrieval": "rgb(59, 130, 246)",
  "SerpAPI":           "rgb(139, 92, 246)",
  "Local Retrieval":   "rgb(107, 114, 128)",
  "All Active Sources":"rgb(14, 165, 233)",
};

export const SourceGrid = memo(function SourceGrid({ documents }: SourceGridProps) {
  if (documents.length === 0) {
    return <div className="empty-state compact">No sources yet.</div>;
  }

  return (
    <div className="source-grid">
      {documents.map((document, idx) => {
        const source =
          typeof document.metadata.source === "string"
            ? document.metadata.source
            : "Unknown";
        const mmrRank =
          typeof document.metadata.mmr_rank === "number"
            ? (document.metadata.mmr_rank as number)
            : idx + 1;
        const sourceColor = SOURCE_COLORS[source] ?? "rgb(107, 114, 128)";
        return (
          <article className="source-card" key={document.id}>
            <div className="source-meta">
              <span>{document.citation}</span>
              <span
                className="score-badge"
                style={{ color: scoreColor(document.score) }}
                title="Relevance score"
              >
                {document.score > 0 ? document.score.toFixed(3) : "—"}
              </span>
              <span
                style={{
                  fontSize: "0.7rem",
                  color: "rgb(148, 163, 184)",
                  fontVariantNumeric: "tabular-nums",
                }}
                title="MMR rank"
              >
                #{mmrRank}
              </span>
            </div>
            <div className="source-tags">
              <span style={{ color: sourceColor, fontWeight: 600, fontSize: "0.7rem" }}>
                {source}
              </span>
            </div>
            {document.url ? (
              <a href={document.url} target="_blank" rel="noreferrer">
                {document.title}
                <ExternalLink size={14} />
              </a>
            ) : (
              <h3>{document.title}</h3>
            )}
            <p>{document.content}</p>
          </article>
        );
      })}
    </div>
  );
});
```

- [ ] **Step 4: Run all frontend tests**

```bash
cd web && npx vitest run src/components/__tests__/SourceGrid.test.tsx
```

Expected: 8 passed (6 existing + 2 new).

```bash
cd web && npm run typecheck && npx vitest run
```

Expected: no type errors, all tests pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/SourceGrid.tsx \
        web/src/components/__tests__/SourceGrid.test.tsx
git commit -m "feat(ui): MMR rank badge and color-coded score in SourceGrid"
```

---

### Task 4: Push and open PR

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/unit/ -v --tb=short 2>&1 | tail -20
cd web && npx vitest run 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 2: Push and open PR**

```bash
git push -u origin feat/browser-hybrid-rerank-mmr
gh pr create \
  --title "feat: browser two-phase retrieval, hybrid search, cross-encoder rerank, MMR UI" \
  --body "$(cat <<'EOF'
## Summary
- **browser.py**: Two-phase retrieval — after getting SERP hit URLs, navigates to each URL with playwright-cli and extracts full page text. Falls back to SERP snippet on timeout/error. Controlled by \`fetch_content: bool = True\` in \`BrowserSearchConfig\`.
- **app.py**: \`SearchExperienceSettings\` gains \`browser_search_url\` (optional running browser.py server) and \`rerank_url\` (optional cross-encoder rerank server). When \`browser_search_url\` is set, browser results are merged with the primary provider before dedup/MMR. When \`rerank_url\` is set, merged docs are sent to the reranker to get real relevance scores before MMR. \`_reindex_documents\` now stamps \`mmr_rank\` on each document.
- **SourceGrid.tsx**: Score badge is color-coded green/yellow/orange/gray by relevance threshold. Source provider label uses distinct colors per backend (browser=blue, serpapi=purple, local=gray). MMR rank number appears as \`#N\` beside the citation.

## Test plan
- [ ] \`pytest tests/unit/retrieval/test_browser_retrieval.py -v\` → 10 passed
- [ ] \`pytest tests/unit/servers/web/ -v\` → all passed including new \`test_reranking.py\`
- [ ] \`cd web && npx vitest run\` → all passed
- [ ] Optional end-to-end: start demo retrieval on 8001, start browser.py on 8002, set \`BROWSER_SEARCH_URL=http://localhost:8002\` env, start web backend on 7860, start frontend on 5173 — submit a query and verify source cards show score colors and MMR rank badges

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

### 1. Spec coverage

| Requirement | Covered |
|-------------|---------|
| playwright: browser.py navigates to result URLs for full content | Task 1 |
| hybrid: browser results merged with other provider results | Task 2 (`_run_browser_search` + `browser_search_url`) |
| reranking: cross-encoder scores before MMR | Task 2 (`_rerank_documents` + `rerank_url`) |
| UI shows results using MMR | Task 2 (`mmr_rank` in `_reindex_documents`) + Task 3 (badge in `SourceGrid`) |

All requirements covered. ✓

### 2. Placeholder scan

No TBDs, TODOs, or "similar to" references. All code blocks are complete. ✓

### 3. Type consistency

- `BrowserSearchConfig.fetch_content: bool` — defined Task 1 step 4, used Task 1 test step 1 ✓
- `_fetch_page_text(url, *, session) -> str` — defined and called in same Task 1 ✓
- `SearchExperienceSettings.browser_search_url: str | None` — defined Task 2 step 3, threaded step 9 ✓
- `SearchExperienceSettings.rerank_url: str | None` — defined Task 2 step 3, threaded step 9 ✓
- `_run_browser_search(query, *, browser_search_url, top_k, existing_count) -> list[ContextDocument]` — defined step 5, called steps 7 and 8 ✓
- `_rerank_documents(docs, query, rerank_url) -> list[ContextDocument]` — defined step 6, called steps 7 and 8, monkeypatched in test ✓
- `ContextDocument` constructed directly (frozen dataclass, no `model_copy`) — consistent throughout ✓
- `document.metadata.mmr_rank` as `number` in TypeScript — set in backend Task 2 step 10, read in Task 3 ✓
- `scoreColor(score: number) -> string` returns RGB string literals — `style={{ color: scoreColor(...) }}` and `badge.style.color` comparison both use RGB ✓
