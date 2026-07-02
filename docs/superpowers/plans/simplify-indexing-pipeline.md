# Plan: Simplify the Indexing Pipeline

Spec: [simplify-indexing-pipeline.md](../specs/simplify-indexing-pipeline.md)
Strategy: extract cohesive modules from the 1,581-LOC `index_builder.py`, keep it
as a back-compat facade, verify tests green after each step. Pure refactor.

Branch: `refactor/simplify-indexing-pipeline` (create before first commit).

## Phase 0 — Baseline & safety net

1. Create the branch. → verify: `git branch --show-current`.
2. Capture baseline: `pytest -q` (record pass/fail counts, note any
   pre-existing failures) and `ruff check .`. → verify: baseline saved; this is
   the "same results after" bar.
3. Grep the exact public symbols imported from `index_builder` across `src/`,
   `tests/`, `examples/`, `docs/`. → verify: a definitive list of names the
   facade must re-export.

## Phase 1 — Extract `faiss_io.py` (lowest coupling)

Move: `set_hnsw_ef_construction/search`, `write_dense_faiss_index`,
`write_faiss_index`, `write_embeddings_memmap`, `write_corpus_jsonl`,
`_Corpus`, `load_corpus`, `load_corpus_from_connector`, `dump_connector_to_jsonl`.

1. Create `faiss_io.py`; cut/paste functions unchanged; keep `_require_faiss`
   lazy guard local or shared. → verify: module imports without torch/faiss at top.
2. Re-export from `index_builder.py`. → verify: `pytest tests/unit/test_indexing_pipeline.py`.
3. `ruff format`. → verify: clean.

## Phase 2 — Extract `chunking.py`

Move: `chunk_document`, `chunk_documents`, `generate_large_chunks`,
`filter_indexable_documents`, `_tokenize_for_chunking`, `_token_count`,
`_extract_blurb`, `_make_mini_chunk_texts`, `_split_*`, `_overlap_tail`,
`_combine_index_chunks`, `_metadata_suffix_for_index`, `_batched`.

1. Create `chunking.py`; move functions unchanged. → verify: no behavior change.
2. **Investigate** whether `_tokenize_for_chunking` duplicates `text.py`
   tokenizers. If byte-identical behavior → reuse `text.py`; else add a one-line
   comment noting they are intentionally distinct. → verify: decision recorded.
3. Re-export from facade. → verify: `pytest tests/unit/test_indexing_pipeline.py
   tests/unit/test_connectors.py`.

## Phase 3 — Extract `embedding.py`

Move: `prepare_texts`, `_apply_text_prefix`, `load_model`, `pooling`,
`_encode_batch`, `embed_chunks`, `embed_chunks_with_failure_handling`,
`_embed_texts`, `_get_title_embedding`, `_embed_chunk_batch`,
`_coerce_embedding_matrix`, `_normalize_embedding_rows`, `resolve_pooling_method`,
`deterministic_embedding_fn`.

1. Create `embedding.py`; keep `_require_torch/_require_transformers/_require_tqdm`
   lazy. → verify: `python -c "import src.internal.document_index.embedding"`
   works without torch installed (CI torch-gap guard).
2. Re-export from facade. → verify: `pytest tests/unit/test_indexing_server_facade.py`
   (imports `deterministic_embedding_fn`).

## Phase 4 — Extract `cli.py`

Move: `IndexBuilderConfig`, `IndexBuilder`, `parse_args`, `main`,
`IndexingHeartbeatInterface` if CLI-only (else keep shared).

1. Create `cli.py`; move the build-tool code. → verify: functions moved.
2. Keep `python -m src.internal.document_index.index_builder` working: facade's
   `main`/`__main__` delegates to `cli.main`. → verify:
   `python3 -m src.internal.document_index.index_builder --help`.
3. Confirm `run_indexing_pipeline` + heartbeat helpers land in the right home
   (`indexing.py` or a small `pipeline.py`) — keep import path stable via facade.

## Phase 5 — Facade cleanup & verification

1. Reduce `index_builder.py` to imports + re-export block (`__all__`). Target
   < 150 LOC. → verify: `wc -l` and grep the Phase-0 symbol list all resolve.
2. Full `pytest` (unit + regression). → verify: identical results to Phase 0
   baseline.
3. `ruff check . --fix && ruff format .`. → verify: clean.
4. Sanity: `git diff --stat` should show moves (roughly balanced +/- across
   new files), not rewrites.

## Phase 6 — Ship

1. Update `CLAUDE.md` architecture note only if the module list is referenced
   there (check first; likely not needed).
2. Commit on the branch, push, open PR referencing this spec + plan.
   → verify: PR created, CI green.

## Risk register

- **Missed re-export** → import error in a test. Caught by Phase 1–5 test runs;
  the import-based tests are the safety net.
- **Circular import** between new modules (e.g. `chunking` ↔ `embedding` via
  shared helpers). Mitigation: put shared low-level helpers in the module that
  owns them and import one-directionally; if a cycle appears, factor the shared
  bit into `faiss_io`/`text.py` or a tiny `_common.py`.
- **Lazy-import regression** re-triggers the CI torch gap. Mitigation: Phase 3
  explicit import-without-torch check.
- **Squash drops a moved file** (known stacked-PR gotcha) — this is a single PR,
  single branch, so low risk; still verify the merged diff post-merge.
