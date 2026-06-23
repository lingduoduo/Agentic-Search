# Demo Corpus Relevance + BEIR Converter — Implementation Plan

Spec: `docs/superpowers/specs/2026-06-23-demo-corpus-relevance-and-beir-converter-design.md`
Branch: `feat/beir-corpus-converter`

## Steps

1. **Reproduce + write failing test** → new `tests/unit/servers/retrieval/test_demo_retrieval.py`
   with a ranking case and `test_retrieve_drops_zero_relevance_documents` (query absent from a
   2-doc corpus must return `[]`).
   Verify: zero-relevance test fails against current code (returns 2 docs at score 0.0).

2. **Fix the relevance filter** → in `src/internal/servers/retrieval/demo.py` `retrieve`,
   filter `enumerate(row_scores)` to `score > 0.0` before `sorted(...)[:topk]`.
   Verify (TDD): both demo tests pass; full `tests/unit/servers/retrieval/` stays green.

3. **Write `scripts/beir_to_corpus.py`** → argparse CLI (`--dataset/--out/--data_dir/--limit`),
   lazy `beir` import with install hint, BEIR→demo record mapping, skip empty docs, print the
   demo-server run command.
   Verify: `python3 scripts/beir_to_corpus.py --dataset nfcorpus` writes 3,633 docs to
   `data/corpus_nfcorpus.jsonl`.

4. **End-to-end check** → load the generated corpus with `TfidfRetriever`; a real query
   (`cholesterol heart disease`) returns ranked docs with nonzero scores; `GRPO` returns `[]`.

5. **Lint** → `ruff check scripts src tests --fix && ruff format` clean.

6. **Commit (spec+plan+code), push, open PR** on `feat/beir-corpus-converter`.

## Risk / rollback

Low risk. The filter only removes docs that scored 0 (no shared terms) — these were never
useful results. The converter is additive and writes only into gitignored `data/`. Rollback =
revert the branch; no migrations, no API/contract changes.
