# Spec: Simplify the Indexing Pipeline

Status: Approved — facade approach
Scope: `src/internal/document_index/` (indexing/write path) + `src/internal/servers/backgroundworker/`
Type: **Pure refactor** — no behavior change, tests stay green.

## 1. Objective

The indexing pipeline (chunk → embed → index into FAISS/BM25, driven by async
background workers) has accreted into a hard-to-navigate monolith. The single
biggest problem: [`index_builder.py`](../../../src/internal/document_index/index_builder.py)
is **1,581 LOC** and mixes at least eight unrelated responsibilities in one file:

| Region (approx. lines) | Responsibility |
|---|---|
| 97–256 | Optional-dep guards, model loading, pooling, batch encoding |
| 256–317 | Corpus loading (`_Corpus`, `load_corpus*`, `dump_connector_to_jsonl`) |
| 317–393 | FAISS index writing + HNSW knobs |
| 393–485, 929–1134 | Chunking / text-splitting subsystem |
| 485–757 | Chunk embedding + failure handling + vector coercion/normalization |
| 758–832 | Artifact writers (jsonl, memmap, faiss) |
| 832–928 | `run_indexing_pipeline` orchestration + heartbeat helpers |
| 1140–1550 | Standalone CLI (`IndexBuilderConfig`, `IndexBuilder`, `parse_args`, `main`) |

**Goal:** split this monolith into cohesive modules along the responsibilities
above, remove verified duplication, and leave every existing import and test
working unchanged. Reduce complexity and file size; change no runtime behavior.

**Target users:** engineers maintaining ingestion (the workers) and the
retrieval indexing code. Success = they can find "where chunking lives" or
"where FAISS gets written" without scrolling a 1.5k-line file.

### Explicitly out of scope

The read/query path is a separate concern and **not** touched here:
- `src/internal/retrieval/` (query transforms, rerankers, eval/BEIR/Ragas,
  fusion learner, caches).
- OpenSearch/Weaviate backend internals (`opensearch/`, `weaviate/`).
- Connector implementations (`src/internal/connectors/`).
- Worker *coordination* redesign (queues/scheduling). We may relocate helpers
  but will not change how workers are scheduled or how batches flow.

## 2. Acceptance criteria

1. `index_builder.py` drops from ~1,581 LOC to a thin facade (target < 150 LOC)
   that re-exports the public names, OR is removed with importers updated —
   decided in the plan. Every symbol currently importable from
   `src.internal.document_index.index_builder` remains importable from the same
   path (back-compat shim) so no caller changes are forced.
2. Responsibilities live in cohesive new modules (proposed):
   - `chunking.py` — `chunk_document`, `chunk_documents`, `generate_large_chunks`,
     `_split_*`, `_combine_index_chunks`, chunk tokenization.
   - `embedding.py` — `load_model`, `pooling`, `prepare_texts`, `_encode_batch`,
     `embed_chunks*`, `_embed_*`, vector coercion/normalization.
   - `faiss_io.py` — `write_dense_faiss_index`, `set_hnsw_*`, `write_faiss_index`,
     `write_embeddings_memmap`, `write_corpus_jsonl`, corpus loading.
   - `cli.py` — `IndexBuilderConfig`, `IndexBuilder`, `parse_args`, `main`
     (the `python -m` build tool).
   - `run_indexing_pipeline` + heartbeat helpers stay with the orchestration
     entry point (likely `indexing.py` or a small `pipeline.py`).
   Exact module names/boundaries are finalized in the plan; this list is the
   intent, not a contract.
3. **No behavior change.** `pytest` passes with the same results before and
   after (unit + regression). No new test failures; no skipped tests newly
   introduced. `ruff check` and `ruff format` clean.
4. Verified duplication is removed; *suspected* duplication is investigated and
   either removed (if truly identical) or documented as intentionally distinct.
   Specifically: confirm whether chunk tokenization (`_tokenize_for_chunking`)
   can reuse `text.py` tokenizers, and whether any splitter helpers overlap.
   Do **not** merge `_normalize_embedding_rows` (vector L2 norm) with text
   normalization — they are different operations.
5. Public package surface (`src/internal/document_index/__init__.py`) exports
   the same names as before.

## 3. Commands

```bash
pip install -e .                      # package importable
pytest                                # unit + regression must stay green
pytest tests/unit/test_indexing_pipeline.py -v
pytest tests/unit/test_connectors.py tests/regression/test_regression.py
ruff check . --fix && ruff format .
# CLI still works after the split:
python3 -m src.internal.document_index.index_builder --help   # or new cli path
```

## 4. Project structure (after)

```
src/internal/document_index/
  index_builder.py     # thin back-compat facade (re-exports) — was 1,581 LOC
  chunking.py          # NEW — text splitting + chunk assembly
  embedding.py         # NEW — model load, pooling, encode, embed chunks
  faiss_io.py          # NEW — FAISS/memmap/jsonl artifact writers + corpus I/O
  cli.py               # NEW — IndexBuilder CLI (parse_args/main)
  indexing.py          # entry points (Chunker, index_documents, ChunkSink) — unchanged surface
  text.py, models.py, interfaces.py, factory.py, ...  # unchanged
```

Module boundaries are the deliverable of the plan; the facade guarantees
importers and tests don't move.

## 5. Code style

- Match existing style; no drive-by reformatting outside moved code.
- Moves are cut/paste of whole functions — preserve signatures, docstrings,
  and behavior byte-for-byte. Only edit import lines needed for the move.
- Keep optional-dependency lazy-import guards (`_require_torch`, `_require_faiss`,
  `_require_transformers`, `_require_tqdm`) intact — CI relies on lazy torch
  import (see `project_ci_torch_gap` memory). Do not import torch/faiss at
  module top level.
- No new abstractions, config objects, or "flexibility" — this is a move, not a
  redesign.

## 6. Testing strategy

- **Baseline first:** capture `pytest` results (counts + any pre-existing
  failures) before touching code. The bar is "same results after."
- After each module extraction, run the indexing-related tests
  (`test_indexing_pipeline.py`, `test_connectors.py`,
  `test_indexing_pipeline_facade.py`, `test_regression.py`) — these directly
  import from `index_builder`.
- Full `pytest` + `ruff` before opening the PR.
- No new tests required (pure refactor), but if a facade re-export is missed the
  existing import-based tests will catch it — that's the safety net.
- Web TestClient tests: run via `examples/run_web_integration_tests.sh` if
  touched (model-load hang gotcha, per `project_web_test_model_load`).

## 7. Boundaries

**Always:**
- Preserve the public import surface via a re-export facade.
- Keep tests green at every step; verify with `pytest` after each extraction.
- Work on a feature branch, open a PR at the end (never commit to main).

**Ask first:**
- Deleting `index_builder.py` outright (vs. keeping it as a facade) and updating
  all importers.
- Any change that alters chunking output, embedding vectors, or index files.
- Relocating helpers *out of* `document_index/` (e.g., into workers).

**Never:**
- Change chunking/embedding/index behavior or output artifacts.
- Touch the retrieval/query path, connectors, or backend internals in this PR.
- Add module-top-level heavy imports (torch/faiss/transformers).
- Introduce new features, config, or abstractions.
```

## Resolved decisions

1. **Facade** (approved) — `index_builder.py` becomes a thin re-export shim;
   the ~11 importers and all tests stay untouched. No hard cutover.
2. **CLI location** — code moves to `cli.py`, but
   `python -m src.internal.document_index.index_builder` keeps working via the
   facade's `main`/`__main__` delegation.
