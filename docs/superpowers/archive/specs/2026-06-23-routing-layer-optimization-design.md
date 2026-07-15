# Spec: Routing Layer — Optimization & Query Construction (M10)

- **Date:** 2026-06-23
- **Status:** Draft (awaiting approval)
- **Branch:** `feat/routing-layer-optimization` (to be created; never commit to `main`)
- **Plan:** `docs/superpowers/plans/2026-06-23-routing-layer-optimization.md` (to be committed on the same branch)
- **Predecessor:** M9 query-transform tuning (`feat/query-transform-optimization-tuning`, PR #319)

## 1. Objective

The query-transform stack (M1–M9) decides **which transforms** to run per query.
It does **not** decide **where a query should go** or **how to express that query
for the chosen backend**. Today retriever selection is *static* (`RETRIEVAL_BACKEND`
env, fixed at process start), there is no domain/source routing, and only one of
the six query-construction targets (metadata filters) exists.

M10 adds the RAG **Routing → Query Construction** stage: a per-query layer that
(1) **routes** a query to a domain, one or more sources, and a retriever strategy,
then (2) **constructs** the backend-specific query for that strategy. All six
construction targets are in scope.

Two layers, both new, both feature-flagged and default-off:

1. **Routing layer** — given a query, produce a `RouteDecision`
   (`domain`, `sources`, `retriever`, `construction_target`, `confidence`). Three
   strategies, mirroring the existing `QueryRouter` pattern: **logical** (structured
   classification into a configured route schema), **semantic** (embedding
   similarity between the query and route descriptions), and a rule-based
   **heuristic fallback**. Backed by a config-driven **route registry**.
2. **Query-construction layer** — a common `QueryConstructor` interface with six
   implementations:
   - **Metadata Filter Construction** — *wraps the existing*
     `src/internal/retrieval/query_constructor.py` (no rewrite).
   - **Vector Search Query Construction** — top_k / namespace / filter / embedding
     params for the dense leg.
   - **Hybrid Retrieval Query Construction** — fusion weights, over-fetch, leg
     selection (wraps existing fusion config).
   - **SQL Query Generation** — net-new, schema-aware Text-to-SQL.
   - **Knowledge Graph Query Construction** — net-new, Cypher/SPARQL templating
     from extracted entities.
   - **API Request Construction** — net-new, NL → structured API request params.

**Backends reality (decided):** only the retrieval backends (local / opensearch /
weaviate) actually exist. SQL DB, knowledge graph, and external APIs are **not
available** in this repo. The three net-new constructors therefore **build and
validate a query object but do not execute it** — correctness is proven by unit
tests (NL → expected SQL/Cypher/request struct), and their executors are
fallback-safe stubs returning empty results so routing to them never breaks a
request. When a real backend is wired later, only the executor changes.

**Target users:** operators of the retrieval stack who enable routing in
production via flags, plus downstream RAG quality. No end-user-facing UI change in
this milestone.

### Non-goals
- No real SQL/KG/API execution backends (interface + tests only for those three).
- No change to default behavior when routing flags are unset (must stay
  zero-overhead; the static `RETRIEVAL_BACKEND` path is untouched when disabled).
- No rewrite of existing modules: `query_constructor.py`, `fusion.py`,
  `query_router.py`, `service.py` are *extended/consumed*, not rewritten. No
  changes to public signatures `transform(...)` / `retrieval_variants(...)`.
- No frontend changes; no swap of embedding/rerank model.
- Not a replacement for the transform-level `QueryRouter` — that picks transforms;
  this picks domain/source/retriever. The two compose.

## 2. Acceptance Criteria (all four gates required)

1. **Routing accuracy** — a new labeled set `data/eval/routing_labels.jsonl`
   (NL query → expected `domain` / `retriever`) scored by a new
   `routing_accuracy` metric. The default router (heuristic, no LLM needed) must
   meet a committed threshold (e.g. ≥ 0.8 top-1 on the labeled set); the metric and
   threshold are recorded in the plan's results note.
2. **Retrieval not regressed** — with routing **enabled** against the demo/served
   index, `python -m src.internal.retrieval.eval_runner --dataset
   data/eval/qa_pairs.jsonl --top_k 10` shows recall@10 / nDCG@10 **≥**
   `data/eval/baseline_metrics.json`. With routing **disabled**, behavior and
   numbers are byte-identical to today (zero-overhead proof).
3. **Construction correctness** — every constructor has unit tests mapping a
   natural-language input to its expected structured output: metadata filter dict,
   vector params, hybrid config, SQL string, Cypher/SPARQL string, API request
   struct. Net-new constructors degrade to a safe empty/None on LLM failure.
4. **Suite green** — full `pytest` stays green (≥ current 2036 tests, no
   regressions); `ruff check . --fix && ruff format .` clean.

A change that improves routing accuracy but regresses the retrieval eval gate, or
vice versa, is rejected.

## 3. Architecture & Scope

### 3.1 New package: `src/internal/routing/`

```
src/internal/routing/
  __init__.py
  route.py            # RouteDecision dataclass, RetrieverTarget enum, Route schema
  registry.py         # RouteRegistry: domains → sources → allowed retrievers (config-driven)
  router.py           # Router: heuristic + logical (LLM) strategies, fallback-safe
  semantic_router.py  # SemanticRouter: embedding-similarity route matching
  routing_factory.py  # build_router_from_env() — assembles strategy chain from flags
  construction/
    __init__.py
    base.py           # QueryConstructor protocol + ConstructedQuery result type
    metadata.py       # wraps existing retrieval/query_constructor.py
    vector.py         # vector-search params constructor
    hybrid.py         # hybrid/fusion config constructor (wraps fusion config)
    sql.py            # net-new Text-to-SQL constructor (interface + validation)
    graph.py          # net-new KG (Cypher/SPARQL) constructor (interface + validation)
    api.py            # net-new API-request constructor (interface + validation)
```

### 3.2 Data model (`route.py`)

- `RetrieverTarget` enum: `SPARSE | DENSE | HYBRID | SQL | GRAPH | API | METADATA`.
- `Route` (frozen dataclass): `name`, `description`, `sources: list[str]`,
  `retriever: RetrieverTarget`. Loaded from the registry.
- `RouteDecision` (frozen dataclass): `domain: str`, `sources: list[str]`,
  `retriever: RetrieverTarget`, `construction_target: RetrieverTarget`,
  `confidence: float`, `strategy: str` (which router produced it).

### 3.3 Route registry (`registry.py`)

Config-driven so domains are **not** hardcoded. Loaded from
`ROUTING_REGISTRY_PATH` (JSON) or a small built-in default mirroring the current
corpus (one `docs` domain → local hybrid). Each entry:
`{name, description, sources[], retriever}`. The registry is the single source of
truth the routers classify into and that construction dispatch reads.

### 3.4 Routers (`router.py`, `semantic_router.py`)

- **Heuristic (default, no LLM):** rule-based mapping from query signals to a
  registered route — structured/aggregation cues ("how many", "average", "count",
  "per ... by ...") → `SQL`; relationship cues ("connected to", "related entities",
  "path between") → `GRAPH`; explicit external-data cues → `API`; else the default
  retrieval route (`HYBRID`/`DENSE`/`SPARSE` per registry). This is the path the
  accuracy gate runs against — zero external dependencies.
- **Logical (optional, `ROUTING_LOGICAL=1`):** LLM structured-output classifier
  that picks a registered route by name; falls back to heuristic on any failure.
- **Semantic (optional, `ROUTING_SEMANTIC=1`):** embed query + route descriptions,
  pick max cosine; reuses the existing dense-embedding helper; falls back to
  heuristic if embeddings unavailable.
- All strategies are **fallback-safe**: any failure degrades to the heuristic, and
  the heuristic always returns a valid registered route.

### 3.5 Construction interface (`construction/base.py`)

```python
class ConstructedQuery:        # frozen dataclass
    target: RetrieverTarget
    payload: dict              # backend-specific (filters / sql / cypher / params)
    text: str | None           # the (possibly rewritten) query text for the leg

class QueryConstructor(Protocol):
    def construct(self, query: str, route: RouteDecision) -> ConstructedQuery: ...
```

Each of the six constructors implements `construct`. Net-new ones validate their
output (e.g. SQL parses, only SELECT, table/column allowlist from a provided
schema) and return an empty payload on failure rather than raising.

### 3.6 Integration (`service.py`, consumed not rewritten)

`RetrievalService` gains an optional routing step gated by `ROUTING_ENABLED`
(default off): build router + registry from env, compute `RouteDecision`, select
the construction target, build the backend query, and dispatch. For retrieval
targets (`SPARSE/DENSE/HYBRID/METADATA`) this drives the existing pipeline; for
`SQL/GRAPH/API` it calls the constructor and a fallback-safe stub executor that
returns `[]` (no backend). **When `ROUTING_ENABLED` is unset, `service.py` runs
exactly as today.**

### 3.7 Eval (`eval_runner.py` extension + new dataset)

- New `routing_accuracy(predictions, labels)` in `eval_metrics.py`.
- New labeled set `data/eval/routing_labels.jsonl`.
- `eval_runner` gains an optional `--routing_eval` mode that scores the router
  against the labeled set and prints top-1 accuracy; the default retrieval eval is
  unchanged.

## 4. Commands

```bash
# Setup
pip install -e . && pip install -r requirements.txt

# Routing-accuracy gate (heuristic router; no LLM needed)
python -m src.internal.retrieval.eval_runner --routing_eval \
  --dataset data/eval/routing_labels.jsonl

# Retrieval no-regression gate (routing enabled vs disabled)
ROUTING_ENABLED=1 python -m src.internal.retrieval.eval_runner \
  --dataset data/eval/qa_pairs.jsonl --top_k 10 --retrieval_url http://localhost:8001/retrieve
python -m src.internal.retrieval.eval_runner \
  --dataset data/eval/qa_pairs.jsonl --top_k 10 --retrieval_url http://localhost:8001/retrieve

# Tests + lint
pytest
pytest tests/unit/routing -v
ruff check . --fix && ruff format .
```

## 5. Project Structure (files)

```
# New
src/internal/routing/**                       # routing layer + construction subpackage
data/eval/routing_labels.jsonl                # labeled routing set
tests/unit/routing/**                          # unit tests for routers + constructors
docs/superpowers/specs|plans/2026-06-23-routing-layer-optimization*  # this spec + plan

# Extended (consumed, not rewritten)
src/internal/retrieval/service.py             # optional ROUTING_ENABLED dispatch step
src/internal/retrieval/eval_runner.py         # --routing_eval mode
src/internal/retrieval/eval_metrics.py        # routing_accuracy metric
src/internal/configs/*                         # ROUTING_* env config (if config dataclass pattern used)

# Reused unchanged (imported by constructors)
src/internal/retrieval/query_constructor.py   # metadata filter backing
src/internal/retrieval/fusion.py              # hybrid config backing
```

## 6. Code Style

- Match existing module idioms: `from __future__ import annotations`, frozen
  dataclasses for config/results, env reads via the local `_bool`/`_env` helpers,
  fallback-safe components (any LLM/embedding failure degrades to heuristic /
  empty, never raises out of `construct`/`route`).
- Routers and constructors mirror the `QueryRouter` / `QueryConstructor` shapes
  already in the codebase (heuristic-first, optional learned/LLM layer on top).
- Surgical edits to `service.py` / `eval_runner.py` — only the routing branch is
  added; existing paths untouched. No drive-by refactors.

Illustrative shape:

```python
@dataclass(frozen=True)
class RouteDecision:
    domain: str
    sources: list[str]
    retriever: RetrieverTarget
    construction_target: RetrieverTarget
    confidence: float
    strategy: str

class Router:
    def route(self, query: str) -> RouteDecision:
        try:
            return self._classify(query)          # logical/semantic if enabled
        except Exception as exc:
            logger.warning("router fallback: %s", exc)
        return self._heuristic(query)             # always returns a valid route
```

## 7. Testing Strategy

- **Unit (primary):** `tests/unit/routing/` — heuristic routing rules per
  `RetrieverTarget`; registry load/default; fallback-safety (force LLM/embedding
  failure → heuristic); each constructor's NL → structured-output mapping; net-new
  validators (SQL is SELECT-only / column-allowlisted, Cypher/SPARQL well-formed,
  API request matches schema); empty-on-failure degradation.
- **Routing-accuracy gate:** `routing_accuracy` over `routing_labels.jsonl`.
- **Retrieval gate:** `eval_runner` recall@10 / nDCG@10 vs baseline, routing on and
  off (off must be byte-identical).
- **Zero-overhead proof:** a test asserting `service.py` skips all routing
  construction when `ROUTING_ENABLED` is unset.
- No new integration dependencies; integration suite untouched.

## 8. Boundaries

**Always**
- Work on `feat/routing-layer-optimization`; create it before the first commit.
- Keep every `ROUTING_*` flag defaulting to disabled (zero overhead when unset).
- Keep all routers/constructors fallback-safe (never raise out of `route`/`construct`).
- Run `pytest` + `ruff` green before the PR; commit spec **and** plan on the branch.
- Open a PR after the work with a unique, specific title.

**Ask first**
- Wiring a *real* SQL/KG/API execution backend (out of scope this milestone).
- Any change to a public signature (`transform`, `retrieval_variants`,
  `RetrievalService.search`, router/constructor protocols once published).
- Adding a heavyweight dependency (e.g. a SQL parser, a graph driver) — prefer
  stdlib/lightweight validation first.
- Changing `data/eval/*` baseline contents.

**Never**
- Commit directly to `main`.
- Rewrite or delete `query_constructor.py`, `fusion.py`, `query_router.py`,
  `service.py`, or unrelated code (extend/consume only).
- Execute generated SQL/Cypher/API requests against any live system in tests.
- Change default-on behavior or the static `RETRIEVAL_BACKEND` path.

## 9. Success Criteria (summary)

- [ ] Routing layer routes a query to domain/source/retriever, default heuristic
      meets the accuracy threshold on the labeled set.
- [ ] All six constructors produce validated structured output, unit-tested;
      three net-new ones are interface+validation only (no live backend).
- [ ] `ROUTING_ENABLED=0` → byte-identical behavior; `=1` → retrieval eval ≥ baseline.
- [ ] Full `pytest` green, `ruff` clean, spec + plan committed on the branch, PR opened.

## 10. Open Questions

- **Phasing:** ship as one PR (large) or split routing layer (Phase A) and the
  three net-new constructors (Phase B) into two PRs? Default assumption: **one
  branch, one PR**, since the constructors are interface-only and small — revisit
  if the diff gets large.
- **Learned router:** include a joblib learned route classifier now (like
  `QueryRouter`) or defer until a real labeled corpus exists? Default: **defer** —
  ship heuristic + optional logical/semantic; learned model is a follow-up.
