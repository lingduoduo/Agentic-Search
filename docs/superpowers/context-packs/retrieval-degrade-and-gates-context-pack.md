# Generated Context Pack

# Retrieval Degrade And Gates

## Sources

- [Specification: 2026-07-11-retrieval-degrade-and-gates-design.md](../specs/2026-07-11-retrieval-degrade-and-gates-design.md)
- [Plan: 2026-07-11-retrieval-degrade-and-gates.md](../plans/2026-07-11-retrieval-degrade-and-gates.md)

## Specification Context

### Verification

Regression tests in `tests/unit/retrieval/test_service.py`:
- R1: fake reranker raising `RerankerTimeoutError` (and a generic `RuntimeError`)
  → `search()` returns pre-rerank order, mode without `+reranked`, no raise.
- R2: only `QT_REWRITE=1` set → `from_env()` builds a non-None pipeline.
- R3: 3 variants where the LAST (original) fails under `QT_FUSION_WEIGHTED=1` →
  weighted fuse is NOT used (falls back). Plus: when an earlier paraphrase fails
  but the original survives, the `1.0` weight tracks the original set by identity.

## Implementation Plan Context

### Overview

Date: 2026-07-11
Spec: `docs/superpowers/specs/2026-07-11-retrieval-degrade-and-gates-design.md`

All changes in `src/internal/retrieval/service.py` + tests in
`tests/unit/retrieval/test_service.py`.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
