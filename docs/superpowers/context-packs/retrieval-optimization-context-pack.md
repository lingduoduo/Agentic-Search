# Generated Context Pack

# Retrieval Optimization PRD — Design Spec

## Sources

- [Specification: 2026-06-18-retrieval-optimization-design.md](../archive/specs/2026-06-18-retrieval-optimization-design.md)

## Specification Context

### Out of Scope

- Cross-encoder or LLM reranking (Reranking PRD)
- Connector ingestion pipeline
- Training new embedding models
- UI changes

---

### 2. Architecture

The optimization layer wraps the existing `RetrievalService` without breaking its interface. All changes are additive or internal.

All new components are **opt-in via env vars** — unset = unchanged M1–M4 behavior.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
