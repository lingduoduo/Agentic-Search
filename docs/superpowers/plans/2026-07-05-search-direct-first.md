# SEARCH Direct-First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route SEARCH-intent queries to direct retrieval first; return immediately when the top document score ≥ a threshold (no LLM), and escalate to the SearchAgentLoop only on weak retrieval.

**Architecture:** Preserve the retriever's score through the direct-search path (it's currently dropped at `SearchPage`), then add a `_run_search_direct_or_escalate` helper that retrieves, thresholds on top score, and either returns the docs or falls back to today's SearchAgentLoop / degraded pipeline. Wire it into the SEARCH branch of `_run_auto_routed`.

**Tech Stack:** Python 3.12, FastAPI, pytest.

## Global Constraints

- Strong retrieval (docs non-empty AND `top_score >= T`) returns docs + a non-LLM `_search_only_answer` summary — NO LLM, no agent loop.
- Weak retrieval escalates: `has_local_model` → `_run_search_agent` (today's SearchAgentLoop); else → `_auto_search_pipeline` (today's degraded fallback). Escalation behavior is unchanged.
- `T` = `float(os.environ.get("AGENTIC_SEARCH_SEARCH_DIRECT_MIN_SCORE", "0.2"))`.
- Only SEARCH dispatch changes; intent routing, `classify_route`, the SearchAgentLoop, and the CHAT path are untouched.
- `record_stage` is a no-op when no capture is active — never break the hot path.
- Run `ruff check <files> --fix && ruff format <files>` before each commit (ruff pre-commit hook; if a commit aborts because the hook reformatted, `git add -A` and re-run the same commit).
- Branch: `feat/search-direct-first` (spec already committed there).

---

### Task 1: Preserve the retrieval score through the direct-search path

**Files:**
- Modify: `src/tools/search.py` (`SearchPage` ~line 60; `from_search_result` ~line 67)
- Modify: `src/internal/servers/web/app.py` (`_documents_from_search_pages` line 2039)
- Test: `tests/unit/test_search_tools.py` (append)

**Interfaces:**
- Consumes: existing `SearchResult` (has `.score: float`), `SearchPage`.
- Produces: `SearchPage.score: float`; `_documents_from_search_pages` maps `page.score` onto `ContextDocument.score`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_search_tools.py`:

```python
def test_search_page_preserves_score_from_result():
    from src.context.search import SearchResult
    from src.tools.search import SearchPage

    result = SearchResult(contents="FAISS body", score=0.42, title="FAISS", url="u")
    page = SearchPage.from_search_result(result)
    assert page.score == 0.42


def test_documents_from_search_pages_maps_score():
    from src.internal.servers.web.app import _documents_from_search_pages
    from src.tools.search import SearchPage

    pages = [SearchPage(title="FAISS", summary="body", url="u", score=0.42)]
    docs = _documents_from_search_pages(pages, source_provider="retrieval", query="FAISS")
    assert docs[0].score == 0.42
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_search_tools.py -k "preserves_score or maps_score" -v`
Expected: FAIL — `SearchPage.__init__() got an unexpected keyword argument 'score'` / `AttributeError: 'SearchPage' object has no attribute 'score'`.

- [ ] **Step 3: Add `score` to `SearchPage` and map it**

In `src/tools/search.py`, add the field to the `SearchPage` dataclass and map it in `from_search_result`:

```python
@dataclass(frozen=True)
class SearchPage:
    title: str = ""
    summary: str = ""
    url: str = ""
    error: str | None = None
    score: float = 0.0

    @classmethod
    def from_search_result(cls, result: SearchResult) -> "SearchPage":
        return cls(
            title=result.title or "",
            summary=_compact_contents(result.contents),
            url=result.url or "",
            score=result.score,
        )
```

In `src/internal/servers/web/app.py`, `_documents_from_search_pages` (line 2039), change `score=0.0` to `score=page.score`:

```python
                url=page.url or None,
                score=page.score,
                metadata={
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_search_tools.py -k "preserves_score or maps_score" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ruff check src/tools/search.py src/internal/servers/web/app.py tests/unit/test_search_tools.py --fix && ruff format src/tools/search.py src/internal/servers/web/app.py tests/unit/test_search_tools.py
git add src/tools/search.py src/internal/servers/web/app.py tests/unit/test_search_tools.py
git commit -m "fix(search): preserve retrieval score through SearchPage into documents"
```

---

### Task 2: `_run_search_direct_or_escalate` + SEARCH dispatch

**Files:**
- Modify: `src/internal/servers/web/app.py` (add helper before `_run_auto_routed` ~line 750; SEARCH branch of `_run_auto_routed` ~line 816)
- Test: `tests/unit/test_execution_fallbacks.py` (append)

**Interfaces:**
- Consumes: `_run_direct_search(query, *, source_provider, search_url, browser_search_url=None, rerank_url=None, top_k) -> list[ContextDocument]`; `_run_search_agent(query, *, manager, tokenizer, search_url, top_k, on_turn=None, on_trace=None) -> tuple`; `_auto_search_pipeline(...)`; `_search_only_answer(label, *, queries, documents, source_provider) -> str`; `_capture.record_stage`; `ContextDocument.score` / `.citation`.
- Produces: `_run_search_direct_or_escalate(query, *, manager, tokenizer, llm, search_url, browser_search_url, rerank_url, top_k, filters, history, source_provider, on_turn) -> tuple` returning `(answer, citations, documents, intent, extra)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_execution_fallbacks.py`:

```python
import asyncio

import src.internal.servers.web.app as web_app
from src.context.models import ContextDocument


def _doc(score: float, i: int = 1) -> ContextDocument:
    return ContextDocument(
        id=f"D{i}", title=f"doc{i}", content="body", url=None, score=score, metadata={}
    )


def _call_direct_or_escalate(monkeypatch, direct_docs, agent_result=None):
    async def _fake_direct(*a, **k):
        return direct_docs

    called = {"agent": False}

    async def _fake_agent(*a, **k):
        called["agent"] = True
        return ("agent answer", ["[D1]"], [_doc(0.9)], "search", {})

    monkeypatch.setattr(web_app, "_run_direct_search", _fake_direct)
    monkeypatch.setattr(web_app, "_run_search_agent", _fake_agent)
    result = asyncio.run(
        web_app._run_search_direct_or_escalate(
            "FAISS",
            manager=object(),
            tokenizer=object(),
            llm=None,
            search_url="http://x/retrieve",
            browser_search_url=None,
            rerank_url=None,
            top_k=5,
            filters=None,
            history=[],
            source_provider="retrieval",
            on_turn=None,
        )
    )
    return result, called


def test_strong_retrieval_returns_direct_without_agent(monkeypatch):
    # top score 0.42 >= default threshold 0.2 → direct, agent loop NOT called.
    (answer, citations, documents, intent, extra), called = _call_direct_or_escalate(
        monkeypatch, [_doc(0.42), _doc(0.1, 2)]
    )
    assert called["agent"] is False
    assert extra["search_mode"] == "direct"
    assert documents[0].score == 0.42
    assert intent == "search"


def test_weak_retrieval_escalates_to_agent(monkeypatch):
    # top score 0.1 < 0.2 → escalate; agent loop IS called.
    (answer, citations, documents, intent, extra), called = _call_direct_or_escalate(
        monkeypatch, [_doc(0.1)]
    )
    assert called["agent"] is True
    assert extra["search_mode"] == "escalated"


def test_empty_retrieval_escalates(monkeypatch):
    (_answer, _c, _d, _i, extra), called = _call_direct_or_escalate(monkeypatch, [])
    assert called["agent"] is True
    assert extra["search_mode"] == "escalated"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_execution_fallbacks.py -k "strong_retrieval or weak_retrieval or empty_retrieval" -v`
Expected: FAIL — `AttributeError: module 'src.internal.servers.web.app' has no attribute '_run_search_direct_or_escalate'`.

- [ ] **Step 3: Add the helper**

In `src/internal/servers/web/app.py`, immediately before `async def _run_auto_routed(` (~line 750), add:

```python
def _search_direct_min_score() -> float:
    import os as _os

    return float(_os.environ.get("AGENTIC_SEARCH_SEARCH_DIRECT_MIN_SCORE", "0.2"))


async def _run_search_direct_or_escalate(
    query: str,
    *,
    manager,
    tokenizer,
    llm,
    search_url: str,
    browser_search_url,
    rerank_url,
    top_k: int,
    filters,
    history: list,
    source_provider: str,
    on_turn=None,
) -> tuple:
    """Direct retrieval first; return docs when strong, else escalate.

    Strong = docs non-empty AND top score >= threshold → return the ranked docs
    with a non-LLM summary (no agent loop). Weak → SearchAgentLoop (local model)
    or the degraded pipeline, preserving today's behavior.
    """
    threshold = _search_direct_min_score()
    documents = await _run_direct_search(
        query,
        source_provider="retrieval",
        search_url=search_url,
        rerank_url=rerank_url,
        top_k=top_k,
    )
    real = [d for d in documents if not d.metadata.get("error")]
    top_score = max((d.score or 0.0 for d in real), default=0.0)
    _capture.record_stage(
        "search",
        "direct_retrieval",
        {
            "query": query,
            "top_k": top_k,
            "top_score": top_score,
            "documents": [
                {"id": d.id, "title": d.title, "score": d.score} for d in real
            ],
        },
    )

    if real and top_score >= threshold:
        _capture.record_stage(
            "search",
            "sufficiency",
            {"mode": "direct", "top_score": top_score, "threshold": threshold},
        )
        answer = _search_only_answer(
            "Direct retrieval",
            queries=[query],
            documents=real,
            source_provider="retrieval",
        )
        return (
            answer,
            [d.citation for d in real],
            real,
            "search",
            {"search_mode": "direct", "top_score": top_score},
        )

    _capture.record_stage(
        "search",
        "sufficiency",
        {"mode": "escalated", "top_score": top_score, "threshold": threshold},
    )
    escalate_extra = {
        "search_mode": "escalated",
        "top_score": top_score,
        "escalate_reason": "weak_retrieval",
    }
    has_local_model = manager is not None and tokenizer is not None
    if has_local_model:
        answer, citations, docs, intent, run_extra = await _run_search_agent(
            query,
            manager=manager,
            tokenizer=tokenizer,
            search_url=search_url,
            top_k=top_k,
            on_turn=on_turn,
            on_trace=None,
        )
        run_extra.update(escalate_extra)
        return answer, citations, docs, intent, run_extra

    escalate_extra["route_degraded"] = "no_local_model"
    return await _auto_search_pipeline(
        query,
        llm=llm,
        search_url=search_url,
        browser_search_url=browser_search_url,
        rerank_url=rerank_url,
        top_k=top_k,
        filters=filters,
        history=history,
        source_provider=source_provider,
        extra=escalate_extra,
    )
```

- [ ] **Step 4: Run the helper tests to verify they pass**

Run: `python -m pytest tests/unit/test_execution_fallbacks.py -k "strong_retrieval or weak_retrieval or empty_retrieval" -v`
Expected: PASS.

- [ ] **Step 5: Wire the helper into the SEARCH branch of `_run_auto_routed`**

In `_run_auto_routed` (~line 816), replace the entire `if strategy is RouteStrategy.SEARCH:` block (the `if has_local_model: _run_search_agent(...) else: _auto_search_pipeline(...)`) with a single call:

```python
    # ---- SEARCH: direct retrieval first, escalate to the agent loop if weak ----
    if strategy is RouteStrategy.SEARCH:
        answer, citations, documents, intent, run_extra = (
            await _run_search_direct_or_escalate(
                query,
                manager=manager,
                tokenizer=tokenizer,
                llm=llm,
                search_url=search_url,
                browser_search_url=browser_search_url,
                rerank_url=rerank_url,
                top_k=top_k,
                filters=filters,
                history=history,
                source_provider=source_provider,
                on_turn=on_turn,
            )
        )
        extra.update(run_extra)
        return answer, citations, documents, intent, extra
```

- [ ] **Step 6: Run the web suite for regressions**

Run: `python -m pytest tests/unit/servers/web/ tests/unit/test_execution_fallbacks.py -q`
Expected: PASS — no regressions. (The existing execution-fallback tests that force SEARCH now route through the helper; because their stubs return no strong docs, they escalate exactly as before.)

- [ ] **Step 7: Commit**

```bash
ruff check src/internal/servers/web/app.py tests/unit/test_execution_fallbacks.py --fix && ruff format src/internal/servers/web/app.py tests/unit/test_execution_fallbacks.py
git add src/internal/servers/web/app.py tests/unit/test_execution_fallbacks.py
git commit -m "feat(search): direct-retrieval-first dispatch, escalate to agent loop if weak"
```

---

## Self-Review

**Spec coverage:** score preservation (SearchPage + `_documents_from_search_pages`) → Task 1. `_run_search_direct_or_escalate` with the direct/strong/weak flow, threshold `AGENTIC_SEARCH_SEARCH_DIRECT_MIN_SCORE` default 0.2, escalation to `_run_search_agent`/`_auto_search_pipeline` preserving degradation → Task 2 Steps 3/5. Observability `record_stage("search", "direct_retrieval"/"sufficiency", …)` → Task 2 Step 3. Testing (strong→no agent, weak→agent, empty→escalate, score preservation) → Tasks 1/2 tests. All spec sections covered.

**Placeholder scan:** every step has concrete code, exact paths, exact commands, expected output. No TBD/TODO.

**Type consistency:** `_run_search_direct_or_escalate(...) -> tuple` returns the canonical `(answer, citations, documents, intent, extra)` used by `_run_auto_routed` (Step 5 unpacks exactly that). `SearchPage.score: float` (Task 1) is read as `page.score` in `_documents_from_search_pages` (Task 1) and surfaces as `ContextDocument.score`, read as `d.score` in the helper (Task 2). `_run_search_agent`/`_auto_search_pipeline`/`_search_only_answer`/`_run_direct_search` are called with their real current signatures (verified against source). `record_stage` uses the module-level `_capture` alias already imported in `app.py`.
