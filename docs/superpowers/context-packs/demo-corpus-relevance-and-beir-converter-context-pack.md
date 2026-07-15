# Generated Context Pack

# Demo Corpus Relevance And Beir Converter

## Sources

- [Specification: 2026-06-23-demo-corpus-relevance-and-beir-converter-design.md](../specs/2026-06-23-demo-corpus-relevance-and-beir-converter-design.md)
- [Plan: 2026-06-23-demo-corpus-relevance-and-beir-converter.md](../plans/2026-06-23-demo-corpus-relevance-and-beir-converter.md)

## Specification Context

### Goal

Make the demo retrieval honest and easy to scale: never surface zero-relevance docs, and
provide a one-command path to a real, multi-thousand-doc corpus.

### Scope

- Modify: `src/internal/servers/retrieval/demo.py` — drop score ≤ 0 docs before slicing top-k.
- New: `tests/unit/servers/retrieval/test_demo_retrieval.py` — ranking + zero-relevance cases.
- New: `scripts/beir_to_corpus.py` — download a BEIR dataset and convert to `corpus.jsonl`.

Out of scope: changing the hybrid server's sparse leg (same latent pattern, not the reported
case — noted as follow-up); pinning `beir` as a hard dependency (kept as a lazy import,
matching `beir_eval.py`); committing generated corpora (`data/` is gitignored).

### Verification

- `pytest tests/unit/servers/retrieval/` green (49 tests).
- `python3 scripts/beir_to_corpus.py --dataset nfcorpus` writes 3,633 docs; demo server loads
  it, a real query returns ranked docs with nonzero scores, and `GRPO` returns `[]`.

## Implementation Plan Context

### Risk / rollback

Low risk. The filter only removes docs that scored 0 (no shared terms) — these were never
useful results. The converter is additive and writes only into gitignored `data/`. Rollback =
revert the branch; no migrations, no API/contract changes.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
