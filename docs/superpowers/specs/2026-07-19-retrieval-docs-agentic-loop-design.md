# Document the agentic search loop, the two local stacks, and the routing layers

**Date:** 2026-07-19
**Status:** Approved

## Problem

`docs/retrieval.md` documents retrieval *services and optimization* thoroughly
(reranking, query-transform, routing/query-construction, tuning endpoints) but is
silent on the parts an investigation of the retrieval subsystem surfaced as the
biggest gaps:

- **The agentic search loop itself.** `SearchAgentLoop`
  (`src/agents/search/search.py`) — the escalation target for a weak direct-gate
  match and the "Agentic" in Agentic-Search — is undocumented: its XML action
  protocol, turn cycle, `[RxQyDz]` citation coordinates, and the sufficiency
  control layer (`SearchResultEvaluator` → `EvidenceJudge` → `LoopController`).
- **The direct-first sufficiency gate.** The exact/fuzzy/semantic tiered gate over
  the rank-1 result (`_direct_gate_decision`, `servers/web/app.py`) that decides
  direct-answer vs. escalate is only alluded to.
- **Two independent local stacks.** The doc's server table lists `demo.py`,
  `hybrid.py`, and `server.py` side by side without saying that
  `demo.py`/`hybrid.py` (sklearn TF-IDF + in-memory e5, `POST /retrieve`) are a
  *separate* stack from `server.py` → `RetrievalService` → `LocalBackend`
  (Lucene BM25 / FAISS, `POST /search`). Readers conflate the two "hybrid"s.
- **Four different "routing" mechanisms.** Intent routing, the provider cascade,
  retriever-target routing, and transform routing are each called "routing" and
  are easily confused.

Two smaller accuracy gaps also exist: nothing documents that `LocalBackend`
filters are post-hoc and skip standard doc keys, or that the optional grounding
verifier only checks `[Dx]` (not tool `[Tx]`) citations.

## Goal

Bring `docs/retrieval.md` up to date with the subsystem as it actually behaves —
**additively**, without rewriting existing prose — so the agent loop, the
request ladder, the stack split, and the routing-layer distinctions are all
documented, and two known caveats are called out.

## Design

Five additive edits to `docs/retrieval.md`, each verified against source before
writing:

1. **Two-stacks callout** after the retrieval-servers table: `demo.py`/`hybrid.py`
   (`/retrieve`, raw dicts, no Java/FAISS) vs. `server.py` →
   `RetrievalService` → `LocalBackend` (`/search`, `RetrievalResult` rows) — and
   that the optimization/reranking/QT/routing layers apply to the latter only.
2. **"The direct-first sufficiency gate"** subsection: the exact/fuzzy/semantic
   tier table + the no-e5-model caveat, in the auto-routed request section.
3. **"The agentic search loop"** section: `SearchAgentLoop` XML action-tag table,
   turn cycle, `[RxQyDz]` citation coordinates, and an "Adaptive budget and the
   sufficiency control layer" subsection covering the three collaborating
   components and the GRPO-reward role of the metrics dict.
4. **"Four routing layers, four jobs"** table in the routing section:
   intent / provider cascade / retriever-target / transform.
5. **Two caveats:** `LocalBackend` post-hoc metadata filters that skip
   `id/title/text/contents/url`; and the grounding verifier's `[Dx]`-only regex.

## Scope / non-goals

- **Docs only.** No code changes. No behavior change.
- **Additive.** Existing sections and prose are left intact (0 deletions).
- Not a duplication/dead-code cleanup — a separate investigation confirmed the
  apparent "dead code" (`HybridRetriever`, `QueryBundle`,
  `synthesize_answer_from_context`) is wired/tested/public and the residual
  `_RRF_K`/`_source_prefix` duplication spans intentionally-separate stacks.
  See `project_retrieval_orchestrators_separate` memory. No consolidation.

## Verification

- Every claim checked against source: `backends/local.py` `_KNOWN_DOC_KEYS`;
  `server.py` `/search` vs `demo.py`/`hybrid.py` `/retrieve`; `grounding.py`
  `\[(D\d+)\]` regex; `loop_controller.py`; `evidence_judge.py`;
  `training/evaluation.py`; `search.py` action tags + `effective_search_limit`.
- Referenced path `src/internal/document_index/FILTER_SEMANTICS.md` exists.
- Markdown tables balanced, headings nest correctly.
- Pre-commit hooks (trailing-whitespace, end-of-file) pass.
