# Generated Context Pack

# Retrieval Optimization PRD — Design Spec

## Sources

- [Specification: 2026-06-18-retrieval-optimization-design.md](../specs/2026-06-18-retrieval-optimization-design.md)

## Specification Context

### Out of Scope

- Cross-encoder or LLM reranking (Reranking PRD)
- Connector ingestion pipeline
- Training new embedding models
- UI changes

---

### 2. Architecture

The optimization layer wraps the existing `RetrievalService` without breaking its interface. All changes are additive or internal.

```
POST /search
     │
     ▼
ResultCache.get(query, filters, top_k)        ← NEW M7: Redis result cache
     │ miss
     ▼
QueryOptimizer.expand(query)                  ← NEW M5: expansion + spell-correct
     │
     ▼
RetrievalService.search(expanded_query, ...)
  ├── BM25 leg: SparseRetriever               ← M5: tuned k1/b, BM25+ option
  │   └── QueryExpander (synonyms, acronyms)  ← M5
  │
  ├── Dense leg: FAISSBackend                 ← M6: IVF-PQ, ef_search tuning
  │   └── EmbeddingBatcher (async)            ← M6
  │
  ├── RRF fusion                              ← M7: learned weights per source
  └── MMR rerank                              ← M7: adaptive λ per intent
     │
     ▼
ResultCache.set(...)                          ← NEW M7
     │
     ▼
SearchResponse + latency/cache metrics
```

All new components are **opt-in via env vars** — unset = unchanged M1–M4 behavior.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
