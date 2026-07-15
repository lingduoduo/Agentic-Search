# SEARCH: direct-retrieval first, escalate if weak — design

## Problem

When a query routes to SEARCH, the dispatch layer runs the **SearchAgentLoop** —
a multi-turn, LLM-planner-driven agent (`planner → search_tool → evidence_judge →
loop_controller → answer_generator`). The planner invokes the LLM **before any
retrieval**, so even a bare one-word lookup like `FAISS` pays for an up-front LLM
turn. On the local 1.5B model (MPS) this loop takes minutes and has crashed the
server. Meanwhile the same query via **direct retrieval** returns in ~50 ms with
zero LLM. The up-front planner LLM is redundant for the common case (a clean
lookup that retrieval answers directly).

## Goal

Route SEARCH-intent queries to **direct retrieval first**. If retrieval is strong
(top document score ≥ a configurable threshold), return the results immediately
with **no LLM**. Only if retrieval is weak escalate to the existing
SearchAgentLoop (or the degraded pipeline when no local model). This makes clean
lookups instant and reserves the slow agentic path for queries retrieval can't
answer.

## Non-goals

- Not changing intent classification (regex/LLM routing) — only what SEARCH
  *does* after it's chosen.
- Not changing the SearchAgentLoop itself or the CHAT path.
- Not adding answer synthesis on the strong path — strong results return ranked
  documents plus a lightweight non-LLM summary.
- Not moving the search agent off the local model (separate concern).

## Flow

```
SEARCH intent
  └─ direct retrieval (provider=retrieval, return_scores=true)     # ~50ms, NO LLM
       ├─ docs non-empty AND top_score ≥ T  → STRONG
       │     → return docs + _search_only_answer summary (no LLM). Done.
       └─ else                               → WEAK → escalate:
             has_local_model → _run_search_agent (today's SearchAgentLoop)
             else            → _auto_search_pipeline (today's degraded fallback)
```

- `T` = `AGENTIC_SEARCH_SEARCH_DIRECT_MIN_SCORE` (float, default `0.2`).
- Strong path: zero LLM, returns ranked documents. Weak path: byte-for-byte
  today's behavior (SearchAgentLoop / degraded), just gated behind weak retrieval.
- Escalation cost is unchanged — the slow path only runs when retrieval genuinely
  can't answer.

## Architecture / touch points

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
     source_provider="retrieval"), [d.citation for d in docs], docs, "search",
     {"search_mode": "direct", "top_score": top})`
   - weak: if `has_local_model` → `_run_search_agent(...)`; else
     `_auto_search_pipeline(...)`; merge `extra` with
     `{"search_mode": "escalated", "top_score": top,
     "escalate_reason": "weak_retrieval"}`.
2. **SEARCH branch of `_run_auto_routed`** (~line 816) — call the new helper
   instead of `_run_search_agent`, passing the collaborators it needs. Preserves
   the existing `has_local_model` degradation (now inside the helper).
3. **Score preservation** (`src/tools/search.py`, retrieval-provider branch of
   `search_tool`, ~line 251) — request `return_scores=true` from the retrieval
   server and map the returned `score` onto the `ContextDocument`. Today the
   direct path leaves `score=None`, which would make the threshold unreadable.
   The demo/hybrid retrievers already compute the score (TF-IDF cosine / RRF);
   this only surfaces it.

## Configuration

`AGENTIC_SEARCH_SEARCH_DIRECT_MIN_SCORE` — float, default `0.2`. Read into
`SearchExperienceSettings` (or read via `os.environ` at the helper, matching the
existing pattern for `AGENTIC_SEARCH_REQUEST_CAPTURE_MAX`). Note the threshold is
retrieval-backend-dependent (TF-IDF cosine vs dense vs RRF scales differ); `0.2`
suits the demo TF-IDF (`FAISS` top ≈ 0.256).

**Backend-specific note:** the default `0.2` is tuned for the demo TF-IDF backend
(cosine scores). The `hybrid.py` RRF backend produces much smaller scores — max
≈ `2/61 ≈ 0.033` with the default `rrf_k=60` — so every SEARCH query there falls
below `0.2` and is always classified weak, silently disabling the direct-first
optimization (it degrades to prior behavior, not a crash). Hybrid deployments
must set `AGENTIC_SEARCH_SEARCH_DIRECT_MIN_SCORE` to roughly `0.015`, or the
feature stays a no-op.

## Observability (Request Inspector)

The direct path runs outside the agent loop, so it emits none of the loop's
capture stages. Add, via the existing `request_capture.record_stage`:
- `record_stage("search", "direct_retrieval", {query, top_k, top_score,
  documents})` — the retrieved docs + scores.
- a decision payload `record_stage("search", "sufficiency",
  {mode: "direct"|"escalated", top_score, threshold})` — so the Inspector shows
  why it went direct vs. escalated.

## Testing

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
  returns without invoking the agent loop (assert on the `search_mode: direct`
  extra / no answer_generator stage).
- Full suite green; ruff clean.

## Success criteria

- A clean lookup (`FAISS`) routed to SEARCH returns ranked documents in well under
  a second with **no LLM call** and no agent loop.
- A weak-retrieval query still escalates to the SearchAgentLoop (unchanged
  behavior), gated by the score threshold.
- The Request Inspector shows the direct-vs-escalate decision and the top score.
- The threshold is configurable; degradation when no local model is preserved.
