# Generated Context Pack

# Retrieval PRD — Design Spec

## Sources

- [Specification: 2026-06-15-retrieval-prd-design.md](../specs/2026-06-15-retrieval-prd-design.md)

## Specification Context

### Out of Scope

Cross-encoder reranking improvements, LLM-based reranking, connector ingestion pipeline changes, UI changes.

---

### 2. Architecture

**Single service, pluggable backend.** One `RetrievalService` process owns all three retrieval modes. The backend is selected at startup via `RETRIEVAL_BACKEND=local|opensearch|weaviate`. The factory pattern in `src/internal/document_index/factory.py` is formalized as the single source of truth.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
