# Generated Context Pack

# Search Direct First

## Sources

- [Specification: 2026-07-05-search-direct-first-design.md](../specs/2026-07-05-search-direct-first-design.md)
- [Plan: 2026-07-05-search-direct-first.md](../plans/2026-07-05-search-direct-first.md)

## Specification Context

### Goal

Route SEARCH-intent queries to **direct retrieval first**. If retrieval is strong
(top document score ≥ a configurable threshold), return the results immediately
with **no LLM**. Only if retrieval is weak escalate to the existing
SearchAgentLoop (or the degraded pipeline when no local model). This makes clean
lookups instant and reserves the slow agentic path for queries retrieval can't
answer.

### Architecture / touch points

All in `src/internal/servers/web/app.py` except the score fix:

1. **New helper `_run_search_direct_or_escalate(query, *, manager, tokenizer,
   llm, search_url, browser_search_url, rerank_url, top_k, filters, history,
   source_provider, on_turn) -> tuple`** — returns the canonical
   `(answer, citations, documents, intent, extra)` tuple. Logic:
   - `docs = await _run_direct_search(query, source_provider="retrieval",
     search_url=..., rerank_url=..., top_k=top_k)`
   - `top = max((d.score or 0.0) for d in docs)` if docs else `0.0`
   - strong (`docs and top >= T`): return
     `(_search_only_answer("Direct retrieval", queries=[query], documents=docs,

…

### Testing

- **Unit — `_run_search_direct_or_escalate`** with a stubbed `_run_direct_search`
  and a spy/monkeypatch on `_run_search_agent`:
  - top_score ≥ T → returns the docs, `_run_search_agent` **not** called.
  - top_score < T → escalates, `_run_search_agent` **is** called.
  - empty docs → escalates.
  - no local model + weak → `_auto_search_pipeline` path.
- **Unit — score preservation**: `search_tool(provider="retrieval")` (or its
  client) returns documents with non-None `score` when the retriever provides one
  (stub the retrieval HTTP response with a `score` field).
- **Integration**: a strong-retrieval SEARCH request through `/api/agent`

…

## Implementation Plan Context

### Global Constraints

- Strong retrieval (docs non-empty AND `top_score >= T`) returns docs + a non-LLM `_search_only_answer` summary — NO LLM, no agent loop.
- Weak retrieval escalates: `has_local_model` → `_run_search_agent` (today's SearchAgentLoop); else → `_auto_search_pipeline` (today's degraded fallback). Escalation behavior is unchanged.
- `T` = `float(os.environ.get("AGENTIC_SEARCH_SEARCH_DIRECT_MIN_SCORE", "0.2"))`.
- Only SEARCH dispatch changes; intent routing, `classify_route`, the SearchAgentLoop, and the CHAT path are untouched.
- `record_stage` is a no-op when no capture is active — never break the hot path.

…

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

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_search_tools.py -k "preserves_score or maps_score" -v`

…

### Task 2: `_run_search_direct_or_escalate` + SEARCH dispatch

**Files:**
- Modify: `src/internal/servers/web/app.py` (add helper before `_run_auto_routed` ~line 750; SEARCH branch of `_run_auto_routed` ~line 816)
- Test: `tests/unit/test_execution_fallbacks.py` (append)

**Interfaces:**
- Consumes: `_run_direct_search(query, *, source_provider, search_url, browser_search_url=None, rerank_url=None, top_k) -> list[ContextDocument]`; `_run_search_agent(query, *, manager, tokenizer, search_url, top_k, on_turn=None, on_trace=None) -> tuple`; `_auto_search_pipeline(...)`; `_search_only_answer(label, *, queries, documents, source_provider) -> str`; `_capture.record_stage`; `ContextDocument.score` / `.citation`.

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
