# SEARCH direct-first: tiered match gate — design

## Problem

The SEARCH direct-first dispatch (`_run_search_direct_or_escalate`, shipped in
`2026-07-05-search-direct-first-design.md`) decides "return direct retrieval vs.
escalate to the agent loop" on a **single score threshold**:

```python
if real and top_score >= threshold:   # AGENTIC_SEARCH_SEARCH_DIRECT_MIN_SCORE, default 0.2
```

That `top_score` is the raw retriever score, and its scale is
**backend-dependent**. `0.2` suits demo TF-IDF cosine but is meaningless for the
`hybrid.py` RRF backend (max ≈ `2/61 ≈ 0.033`), where *every* query falls below
the bar and the direct-first optimization silently becomes a no-op. The gate is
a single opaque number that doesn't generalize across retrieval backends.

## Goal

Replace the single score threshold with a **tiered match cascade** between the
query and the rank-1 retrieval result. Each tier is backend-independent (string
ops or a fixed-scale cosine), so the gate behaves identically on TF-IDF, RRF, or
dense. The cascade is biased toward the cheap no-LLM `direct_retrieval` path:
any tier that fires routes direct; only when none fire do we escalate to search.

## Non-goals

- Not changing intent classification, the SearchAgentLoop, or the CHAT path.
- Not changing retrieval, escalation, the `explicit_source` bypass, the
  `try/except` guard, or `_search_only_answer` — only the one strong/weak
  decision line inside `_run_search_direct_or_escalate` changes.
- Not matching the query against a large canonical-term index (no FAISS/ANN —
  the gate is a 1-vs-1 comparison against the top result).
- Not re-embedding full document content (that duplicates the retrieval server).

## The gate

`R` = the rank-1 retrieval result (after dedupe/rerank/MMR). `norm(x)` =
lowercase, strip surrounding whitespace/punctuation, collapse internal
whitespace. Short-circuit, cheapest tier first:

```
_direct_gate_decision(query, docs) -> (is_strong, tier, top_score, cosine)

  1. EXACT      norm(query) == norm(R.title)                → direct   ("exact")
  2. FUZZY      levenshtein(norm(query), norm(R.title)) < 2
                   AND semantic-confirm (cos > COS_MIN)      → direct   ("fuzzy")
                   else                                      → search
  3. SEMANTIC   cos(e5(query), e5(R.title + " " + R.snippet)) > COS_MIN
                                                             → direct   ("semantic")
                   else                                      → search
  4. ELSE       (no tier fired)                              → search   ("weak")
```

- **Tier 1 — exact.** A 100%-normalized match of the query to the top result's
  title. Free. Handles the clean-lookup case (`FAISS` → doc "FAISS"). Instant
  direct, no embedder touched.
- **Tier 2 — fuzzy.** Levenshtein < 2 catches single-char typos
  (`FAISZ`→`FAISS`). Because edit distance 1 also matches unrelated words
  (`cat`→`car`), a near-match is a *candidate* only: it goes direct **iff it is
  also semantically confirmed** by the Tier-3 cosine (> `COS_MIN`); otherwise it
  falls through to `search`. This is the "verify" step — with no LLM available,
  semantic confirmation is the strongest cheap verification.
- **Tier 3 — semantic.** Real Sentence-BERT (e5) cosine on a fixed [-1, 1]
  scale, so the `> 0.8` threshold means the same thing on every backend. Query
  embedded as `"query: <q>"`, the result as `"passage: <title + snippet>"`
  (e5 prefix convention); both L2-normalized; cosine = dot product.
- **Tier 4 — else.** Nothing matched → escalate to search. There is **no**
  score-threshold fallback; the old `top_score >= 0.2` gate is removed entirely.

`is_strong` true → return direct docs (unchanged `_search_only_answer`, no LLM).
`is_strong` false → today's `_escalate(...)` (SearchAgentLoop or degraded
pipeline), byte-for-byte unchanged.

## Architecture / touch points

1. **New pure helper `_direct_gate_decision(query, docs, *, cos_min, embedder)
   -> (is_strong: bool, tier: str, top_score: float, cosine: float | None)`**
   in `src/internal/servers/web/app.py`. No I/O except the injected `embedder`
   call in Tiers 2/3. Fully unit-testable with a fake `EmbeddingFn`.
2. **`_run_search_direct_or_escalate`** (`app.py:857`) — replace
   `if real and top_score >= threshold:` with
   `is_strong, tier, top_score, cosine = _direct_gate_decision(query, real, ...)`
   then `if is_strong:`. The strong-return `extra` gains `"tier"`; the escalate
   path is unchanged.
3. **Levenshtein.** Small internal helper (bounded early-exit at distance 2 — we
   only care about `< 2`), no new dependency.
4. **Embedder.** Reuse the existing `EmbeddingFn` protocol
   (`src/internal/document_index/indexing.py`) and the `intfloat/e5-base-v2`
   loader already used elsewhere. A **lazy module-level singleton**, warmed once
   in the FastAPI lifespan. If `sentence-transformers`/torch is unavailable or
   the load throws, the embedder is `None`: Tier 3 is skipped and Tier 2 cannot
   semantically confirm, so both fall through to `search`. The hot path never
   crashes on a missing model (same philosophy as the existing `try/except`).

## Configuration

- `AGENTIC_SEARCH_SEARCH_DIRECT_COS_MIN` — float, default `0.8`. Cosine bar for
  Tiers 2/3.
- Edit-distance bound (`< 2`) is fixed in code.
- `AGENTIC_SEARCH_SEARCH_DIRECT_MIN_SCORE` is **removed** from the gate (the
  tiers replace it). The env var / `_search_direct_min_score` helper is deleted
  if nothing else reads it.

## Observability (Request Inspector)

The `direct_retrieval` capture stage is unchanged. The `sufficiency` stage gains
`tier` (`exact`/`fuzzy`/`semantic`/`weak`) and `cosine`, so the Inspector shows
*which* rung routed the query direct vs. to search, and the measured cosine.

## Testing

- **Unit — `_direct_gate_decision`** with an injected deterministic
  `EmbeddingFn` (repo has `deterministic_embedding_fn` / `numpy_embedding_fn`),
  never a real model:
  - exact title match → direct, tier `exact`, embedder **not** called.
  - typo (Levenshtein 1) + high cosine → direct, tier `fuzzy`.
  - typo (Levenshtein 1) + low cosine (`cat`/`car`) → search.
  - no lexical match + cosine > 0.8 → direct, tier `semantic`.
  - no lexical match + cosine ≤ 0.8 → search, tier `weak`.
  - empty docs → search.
  - embedder `None` (model unavailable) + only-semantic-would-match → search,
    no crash.
- **Unit — backend independence:** the same query/result pair routes `exact`
  regardless of `top_score` magnitude (proves the RRF no-op is gone).
- **Integration:** a clean-lookup SEARCH request through `/api/agent` returns
  `search_mode: direct`, `tier: exact`, with no agent-loop stage.
- Full suite green; ruff clean.

## Success criteria

- `FAISS` routed to SEARCH returns direct in well under a second, tier `exact`,
  no LLM, **on any retrieval backend** (TF-IDF, RRF, dense) — the backend-
  dependent no-op is gone.
- A typo of a known term goes direct only when semantically confirmed;
  `cat`-for-`car`-style spurious near-matches escalate to search.
- A semantically-equivalent phrasing (cos > 0.8) goes direct; everything else
  escalates, preserving the agentic path for genuinely hard queries.
- Missing embedding model degrades to the lexical tiers + search, never a crash.
- The Request Inspector shows the deciding tier and the cosine.
