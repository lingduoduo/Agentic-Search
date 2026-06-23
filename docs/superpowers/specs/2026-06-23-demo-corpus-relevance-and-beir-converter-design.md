# Demo Retrieval — Zero-Relevance Fix + BEIR Corpus Converter

**Date:** 2026-06-23
**Status:** Approved

## Problem

Two issues surfaced from the same symptom: typing a query absent from the demo corpus
(e.g. `GRPO`) returned 5 unrelated source cards.

1. **Zero-relevance results.** `TfidfRetriever.retrieve` in
   `src/internal/servers/retrieval/demo.py` sorts documents by cosine similarity and slices
   the top-k *without checking the score is nonzero*. `GRPO` shares no terms with any of the
   30 corpus docs, so every score is `0.0`, and the slice returns the first 5 docs in corpus
   order — surfaced as "results" despite zero relevance.

2. **Tiny corpus.** `data/corpus.jsonl` ships only 30 hand-written docs, so most real queries
   have nothing relevant to retrieve. There is no tooling to obtain a larger corpus, even
   though the repo already depends on BEIR for eval (`src/internal/retrieval/beir_eval.py`).

## Goal

Make the demo retrieval honest and easy to scale: never surface zero-relevance docs, and
provide a one-command path to a real, multi-thousand-doc corpus.

## Scope

- Modify: `src/internal/servers/retrieval/demo.py` — drop score ≤ 0 docs before slicing top-k.
- New: `tests/unit/servers/retrieval/test_demo_retrieval.py` — ranking + zero-relevance cases.
- New: `scripts/beir_to_corpus.py` — download a BEIR dataset and convert to `corpus.jsonl`.

Out of scope: changing the hybrid server's sparse leg (same latent pattern, not the reported
case — noted as follow-up); pinning `beir` as a hard dependency (kept as a lazy import,
matching `beir_eval.py`); committing generated corpora (`data/` is gitignored).

## Design

**Relevance filter.** In `retrieve`, filter `enumerate(row_scores)` to `score > 0.0` before
sorting/slicing. When all scores are zero the row becomes `[]`, so the `/retrieve` endpoint
returns no documents and the UI shows no sources instead of filler. Real queries are
unaffected — nonzero-scoring docs rank exactly as before.

**Converter.** BEIR's `GenericDataLoader` returns `corpus[doc_id] = {"title", "text"}`, which
maps 1:1 onto the demo record `{"id", "title", "contents", "metadata": {"acl": ["public"]}}`.
The script downloads (and caches under `--data_dir`) the requested dataset, skips empty docs,
supports an optional `--limit`, and prints the demo-server command to run next. `beir` is
imported lazily with an actionable install message, so the script adds no import-time
dependency.

## Verification

- `pytest tests/unit/servers/retrieval/` green (49 tests).
- `python3 scripts/beir_to_corpus.py --dataset nfcorpus` writes 3,633 docs; demo server loads
  it, a real query returns ranked docs with nonzero scores, and `GRPO` returns `[]`.
