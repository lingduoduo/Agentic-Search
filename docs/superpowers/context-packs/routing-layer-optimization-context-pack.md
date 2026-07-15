# Generated Context Pack

# Routing Layer Optimization

## Sources

- [Specification: 2026-06-23-routing-layer-optimization-design.md](../specs/2026-06-23-routing-layer-optimization-design.md)
- [Plan: 2026-06-23-routing-layer-optimization.md](../plans/2026-06-23-routing-layer-optimization.md)

## Specification Context

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

### 2. Acceptance Criteria (all four gates required)

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

### Tests + lint

pytest
pytest tests/unit/routing -v
ruff check . --fix && ruff format .
```

### 7. Testing Strategy

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

### 10. Open Questions

- **Phasing:** ship as one PR (large) or split routing layer (Phase A) and the
  three net-new constructors (Phase B) into two PRs? Default assumption: **one
  branch, one PR**, since the constructors are interface-only and small — revisit
  if the diff gets large.
- **Learned router:** include a joblib learned route classifier now (like
  `QueryRouter`) or defer until a real labeled corpus exists? Default: **defer** —
  ship heuristic + optional logical/semantic; learned model is a follow-up.

## Implementation Plan Context

### Global Constraints

- Never commit to `main`. Work on `feat/routing-layer-optimization` (already created).
- Every `ROUTING_*` env flag MUST default to disabled (zero overhead when unset).
- New files are allowed (the three net-new constructors need them) but existing modules — `query_constructor.py`, `fusion.py`, `query_router.py`, `service.py`, `eval_runner.py`, `eval_metrics.py` — are **extended/consumed, never rewritten**. No changes to public signatures `transform(...)` / `retrieval_variants(...)` / `RetrievalService.search(...)`.
- Fallback-safe: any LLM/embedding failure in a router or constructor degrades to the heuristic / an empty payload; never raises out of `route()` / `construct()`.
- No real SQL/KG/API execution backends. Net-new constructors build + validate a query object only; executors are stubs returning `[]`.
- No executing generated SQL/Cypher/API requests against any live system, including in tests.
- Acceptance gates (all four): `routing_accuracy` ≥ committed threshold on `data/eval/routing_labels.jsonl`; `eval_runner` recall@10/nDCG@10 ≥ `data/eval/baseline_metrics.json` with routing on, byte-identical off; every constructor unit-tested; full `pytest` green (≥ current 2036), no regressions.
- `ruff check . --fix && ruff format .` clean before each commit.

---

### Task 1: Route data model + registry

The vocabulary every other task imports: target enum, `Route`, `RouteDecision`, and a config-driven registry of domains.

**Files:**
- Create: `src/internal/routing/__init__.py`
- Create: `src/internal/routing/route.py`
- Create: `src/internal/routing/registry.py`
- Test: `tests/unit/routing/__init__.py`, `tests/unit/routing/test_route.py`, `tests/unit/routing/test_registry.py`

**Interfaces:**
- Produces:
  - `RetrieverTarget(str, Enum)` with members `SPARSE, DENSE, HYBRID, METADATA, SQL, GRAPH, API`.
  - `Route(name: str, description: str, sources: tuple[str, ...], retriever: RetrieverTarget)` — frozen.
  - `RouteDecision(domain: str, sources: list[str], retriever: RetrieverTarget, construction_target: RetrieverTarget, confidence: float = 1.0, strategy: str = "heuristic")` — frozen.
  - `RouteRegistry` with `.routes -> list[Route]`, `.get(name) -> Route | None`, `.default() -> Route`, `.by_retriever(t: RetrieverTarget) -> Route | None`, `RouteRegistry.from_env()`, `RouteRegistry.from_file(path)`, and `DEFAULT_ROUTES`.

- [ ] **Step 1: Create the test package init**

Create `tests/unit/routing/__init__.py` (empty file).

- [ ] **Step 2: Write the failing route-model test**

Create `tests/unit/routing/test_route.py`:

```python
from src.internal.routing.route import RetrieverTarget, Route, RouteDecision


def test_retriever_target_values():
    assert RetrieverTarget.SQL.value == "sql"
    assert {t.value for t in RetrieverTarget} == {
        "sparse", "dense", "hybrid", "metadata", "sql", "graph", "api"
    }


def test_route_is_frozen():

_[Section compacted.]_

### Task 2: Heuristic Router + factory

The default routing strategy — rule-based, zero external dependencies, the path the accuracy gate runs against. Plus an env-driven factory.

**Files:**
- Create: `src/internal/routing/router.py`
- Create: `src/internal/routing/routing_factory.py`
- Test: `tests/unit/routing/test_router.py`, `tests/unit/routing/test_routing_factory.py`

**Interfaces:**
- Consumes: `RetrieverTarget`, `Route`, `RouteDecision` (Task 1); `RouteRegistry` (Task 1).
- Produces:
  - `Router(registry: RouteRegistry, llm=None, embedder=None, logical: bool=False, semantic: bool=False)` with `.route(query: str) -> RouteDecision` and `._heuristic(query: str) -> RouteDecision`.
  - `build_router_from_env() -> Router | None` (returns `None` when `ROUTING_ENABLED` is unset).

- [ ] **Step 1: Write the failing heuristic-router test**

Create `tests/unit/routing/test_router.py`:

```python
from src.internal.routing.registry import DEFAULT_ROUTES, RouteRegistry
from src.internal.routing.route import RetrieverTarget
from src.internal.routing.router import Router


def _router():
    return Router(RouteRegistry(DEFAULT_ROUTES))


def test_aggregation_query_routes_to_sql():
    d = _router().route("how many papers were published per year")
    assert d.retriever is RetrieverTarget.SQL
    assert d.domain == "structured"


def test_relationship_query_routes_to_graph():
    d = _router().route("what entities are connected to FAISS")
    assert d.retriever is RetrieverTarget.GRAPH


def test_live_query_routes_to_api():
    d = _router().route("what is the current price of an A100 GPU right now")

_[Section compacted.]_

### Task 3: Logical + semantic router strategies

Optional LLM-backed strategies layered on top of the heuristic. Both fall back to the heuristic on any failure.

**Files:**
- Modify: `src/internal/routing/router.py` (`_logical_route`, `_semantic_route`)
- Create: `src/internal/routing/semantic_router.py`
- Test: `tests/unit/routing/test_router_strategies.py`

**Interfaces:**
- Consumes: `Router`, `RouteRegistry`, `RouteDecision` (Tasks 1–2); an LLM with `.complete(messages) -> LLMResponse | str` (see `src/context/models.py`); an embedder callable `embed(list[str]) -> list[list[float]]`.
- Produces: `cosine_route(query, routes, embedder) -> tuple[Route, float]` in `semantic_router.py`; working `_logical_route` / `_semantic_route` on `Router`.

- [ ] **Step 1: Write the failing strategy tests**

Create `tests/unit/routing/test_router_strategies.py`:

```python
from src.internal.routing.registry import DEFAULT_ROUTES, RouteRegistry
from src.internal.routing.route import RetrieverTarget
from src.internal.routing.router import Router


class _StubLLM:
    def __init__(self, reply):
        self._reply = reply

    def complete(self, messages, **kwargs):
        return self._reply


def test_logical_route_picks_named_route():
    router = Router(RouteRegistry(DEFAULT_ROUTES), llm=_StubLLM("structured"), logical=True)
    d = router.route("break down the dataset somehow")
    assert d.domain == "structured"
    assert d.retriever is RetrieverTarget.SQL
    assert d.strategy == "logical"


def test_logical_route_falls_back_to_heuristic_on_bad_label():

_[Section compacted.]_

### Task 4: Construction interface + existing-backed constructors (metadata, vector, hybrid)

The `QueryConstructor` protocol and the three constructors that wrap code already in the repo.

**Files:**
- Create: `src/internal/routing/construction/__init__.py`
- Create: `src/internal/routing/construction/base.py`
- Create: `src/internal/routing/construction/metadata.py`
- Create: `src/internal/routing/construction/vector.py`
- Create: `src/internal/routing/construction/hybrid.py`
- Test: `tests/unit/routing/test_construction_existing.py`

**Interfaces:**
- Consumes: existing `src.internal.retrieval.query_constructor.QueryConstructor` (`.extract_filters(query) -> tuple[str, dict]`); existing `src.internal.retrieval.fusion_learner.adaptive_mmr_lambda(query) -> float`; `RouteDecision`, `RetrieverTarget`.
- Produces:
  - `ConstructedQuery(target: RetrieverTarget, payload: dict, text: str | None = None)` — frozen.
  - `QueryConstructor` Protocol with `construct(query: str, route: RouteDecision) -> ConstructedQuery`.
  - `MetadataFilterConstructor(llm)`, `VectorSearchQueryConstructor(top_k=10)`, `HybridRetrievalQueryConstructor(rrf_k=60, w_sparse=0.5, w_dense=0.5)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/routing/test_construction_existing.py`:

```python
from src.internal.routing.construction.base import ConstructedQuery
from src.internal.routing.construction.hybrid import HybridRetrievalQueryConstructor
from src.internal.routing.construction.metadata import MetadataFilterConstructor
from src.internal.routing.construction.vector import VectorSearchQueryConstructor
from src.internal.routing.route import RetrieverTarget, RouteDecision

_[Section compacted.]_

### Task 5: SQL Query Generation (net-new, validated, no execution)

Schema-aware Text-to-SQL. The LLM generates; a stdlib validator enforces SELECT-only and a table/column allowlist. No DB is touched.

**Files:**
- Create: `src/internal/routing/construction/sql.py`
- Test: `tests/unit/routing/test_construction_sql.py`

**Interfaces:**
- Consumes: LLM `.complete(messages) -> LLMResponse | str`; `RouteDecision`, `RetrieverTarget`, `ConstructedQuery`.
- Produces:
  - `TableSchema(name: str, columns: tuple[str, ...])` — frozen.
  - `validate_sql(sql: str, schema: list[TableSchema]) -> bool`.
  - `SqlQueryConstructor(llm, schema: list[TableSchema])` with `construct(...)`. On invalid/failed generation, payload is `{"sql": None, "error": "..."}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/routing/test_construction_sql.py`:

```python
from src.internal.routing.construction.sql import (
    SqlQueryConstructor,
    TableSchema,
    validate_sql,
)
from src.internal.routing.route import RetrieverTarget, RouteDecision

_SCHEMA = [TableSchema("papers", ("id", "title", "year"))]


def _route():
    return RouteDecision(
        domain="structured",
        sources=["analytics_db"],
        retriever=RetrieverTarget.SQL,
        construction_target=RetrieverTarget.SQL,
    )


class _StubLLM:
    def __init__(self, sql):
        self._sql = sql

    def complete(self, messages, **kwargs):
        return self._sql


def test_validate_accepts_select_on_known_table():
    assert validate_sql("SELECT year, COUNT(*) FROM papers GROUP BY year", _SCHEMA)


def test_validate_rejects_non_select():

_[Section compacted.]_

### Task 6: Knowledge Graph Query Construction (net-new, validated, no execution)

LLM extracts (entity, relation); a Cypher template is built and validated read-only (MATCH/RETURN, no writes). No graph driver is touched.

**Files:**
- Create: `src/internal/routing/construction/graph.py`
- Test: `tests/unit/routing/test_construction_graph.py`

**Interfaces:**
- Consumes: LLM `.complete(...)`; `RouteDecision`, `RetrieverTarget`, `ConstructedQuery`.
- Produces: `validate_cypher(cypher: str) -> bool`; `KnowledgeGraphQueryConstructor(llm)` with `construct(...)`; payload `{"cypher": str | None, "entity": str | None}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/routing/test_construction_graph.py`:

```python
from src.internal.routing.construction.graph import (
    KnowledgeGraphQueryConstructor,
    validate_cypher,
)
from src.internal.routing.route import RetrieverTarget, RouteDecision


def _route():
    return RouteDecision(
        domain="graph",
        sources=["knowledge_graph"],
        retriever=RetrieverTarget.GRAPH,
        construction_target=RetrieverTarget.GRAPH,
    )


class _StubLLM:
    def __init__(self, reply):
        self._reply = reply

    def complete(self, messages, **kwargs):
        return self._reply


def test_validate_accepts_match_return():
    assert validate_cypher('MATCH (n {name: "FAISS"})-[r]-(m) RETURN n, r, m')


def test_validate_rejects_writes():
    assert not validate_cypher('CREATE (n:Node {name: "x"}) RETURN n')
    assert not validate_cypher('MATCH (n) DETACH DELETE n')


def test_constructor_builds_cypher_from_entity():

_[Section compacted.]_

### Task 7: API Request Construction (net-new, validated, no execution)

LLM extracts request parameters; output is filtered to an allowlisted parameter set for a declared API spec. No HTTP request is made.

**Files:**
- Create: `src/internal/routing/construction/api.py`
- Test: `tests/unit/routing/test_construction_api.py`

**Interfaces:**
- Consumes: LLM `.complete(...)`; `RouteDecision`, `RetrieverTarget`, `ConstructedQuery`.
- Produces: `ApiSpec(name: str, base_url: str, params: tuple[str, ...])` — frozen; `ApiRequestConstructor(llm, spec: ApiSpec)` with `construct(...)`; payload `{"endpoint": str, "params": dict}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/routing/test_construction_api.py`:

```python
from src.internal.routing.construction.api import ApiRequestConstructor, ApiSpec
from src.internal.routing.route import RetrieverTarget, RouteDecision

_SPEC = ApiSpec("prices", "https://api.example.com/prices", ("symbol", "currency"))


def _route():
    return RouteDecision(
        domain="live",
        sources=["external_api"],
        retriever=RetrieverTarget.API,
        construction_target=RetrieverTarget.API,
    )


class _StubLLM:
    def __init__(self, reply):
        self._reply = reply

    def complete(self, messages, **kwargs):
        return self._reply


def test_constructor_extracts_allowlisted_params():
    llm = _StubLLM('{"symbol": "A100", "currency": "USD", "evil": "drop"}')
    out = ApiRequestConstructor(llm, _SPEC).construct("price of A100 in USD", _route())
    assert out.target is RetrieverTarget.API
    assert out.payload["endpoint"] == "https://api.example.com/prices"

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
