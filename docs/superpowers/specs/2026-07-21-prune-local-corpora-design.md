# Spec: Prune unused local corpora + stale registry entry

## Problem
The local `data/` dir accumulated large regeneratable artifacts (BEIR nfcorpus
corpus, NQ/bamboogle training parquet, prebuilt indexes) that bloat the working
tree. All are gitignored and reproducible from their generator scripts. The
tracked `corpora.json` registry still advertises `nfcorpus`, whose backing file
is being removed.

## Scope
Delete (local, untracked — no repo delta):
- `data/corpus_nfcorpus.jsonl` — regen via `examples/beir_to_corpus.py --dataset nfcorpus`
- `data/nq_search/`, `data/bamboogle_train/` — training parquet
- `data/indexes/` — FAISS/BM25, rebuilt on demand

Repo change: remove the `nfcorpus` entry from `data/corpora.json`.

## Kept
- `data/corpus.jsonl` (demo, tracked), `data/corpus_scifact.jsonl` (in use),
  `data/eval/`, `data/intent_examples.json`, and the `demo`/`scifact` registry entries.

## Success criteria
- `data/corpora.json` parses and lists only `demo`, `scifact`.
- No code references the removed `nfcorpus` registry key.
