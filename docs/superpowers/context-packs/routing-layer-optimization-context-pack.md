# Generated Context Pack

# Routing Layer Optimization

## Sources

- [Specification: 2026-06-23-routing-layer-optimization-design.md](../specs/2026-06-23-routing-layer-optimization-design.md)
- [Plan: 2026-06-23-routing-layer-optimization.md](../plans/2026-06-23-routing-layer-optimization.md)

## Specification Context

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

…

## Implementation Plan Context

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

…

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

…

### Task 3: Logical + semantic router strategies

Optional LLM-backed strategies layered on top of the heuristic. Both fall back to the heuristic on any failure.

**Files:**
- Modify: `src/internal/routing/router.py` (`_logical_route`, `_semantic_route`)
- Create: `src/internal/routing/semantic_router.py`
- Test: `tests/unit/routing/test_router_strategies.py`

**Interfaces:**
- Consumes: `Router`, `RouteRegistry`, `RouteDecision` (Tasks 1–2); an LLM with `.complete(messages) -> LLMResponse | str` (see `src/context/models.py`); an embedder callable `embed(list[str]) -> list[list[float]]`.

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
