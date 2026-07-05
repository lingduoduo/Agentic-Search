# Local RAG Smoke-Test Workflow Design

## Goal

Provide a small, offline workflow that proves the repository can retrieve context and produce compact RAG training parquet files using the existing `data/corpus.jsonl`. Keep the current FlashRAG/Natural Questions preparation workflow available, but label it as an optional large-dataset path with external prerequisites.

## Scope

The workflow will add `examples.prepare_local_rag_smoke_dataset`. It will use a small set of bundled question-and-answer examples whose answers are present in the existing demo corpus, retrieve passages with the repository's lightweight TF-IDF retriever, convert each example with the existing RAG training-record helpers, and either preview or write the resulting records.

It will not download datasets, build a production retrieval index, generate Natural Questions caches, modify the existing corpus, or replace `examples.prepare_search_rag_dataset`.

## Architecture and Data Flow

The new example module will contain a small immutable collection of QA examples covering documents in `data/corpus.jsonl`. Each example will contain a question and one or more accepted answers.

At runtime the command will:

1. Validate and load the requested corpus path.
2. Construct the existing `TfidfRetriever` directly; no HTTP server is required.
3. Retrieve the requested top-k documents for every bundled question.
4. Fail if a question has no retrieved documents rather than producing an empty-context training record.
5. Convert retrieval results into the context format expected by `build_search_rag_record`.
6. Print auditable records in preview mode or write a compact parquet dataset in output mode.

The command will default to `data/corpus.jsonl`, a small top-k value, and an output path under `data/`. CLI arguments will allow callers to override the corpus, top-k, output path, and preview behavior.

## Component Boundaries

- `TfidfRetriever` remains responsible only for ranking corpus documents.
- The new example module owns the bundled QA examples, result adaptation, validation, preview rendering, and parquet orchestration.
- Existing functions in `src.training.data` remain responsible for canonical RAG prompt and reward-record formatting.
- `examples.prepare_search_rag_dataset` remains the optional adapter for FlashRAG/NQ plus externally generated retrieval caches.

The smoke workflow will import and reuse these components instead of duplicating retrieval or training-record logic.

## Output Contract

Each output row will retain the repository's compact RAG schema:

- `data_source`
- `prompt`, containing the question and retrieved context
- `ability`
- `reward_model`, containing accepted answers
- `extra_info`, containing split and row index

Preview output will show the question, accepted answers, retrieved document identifiers or context excerpt, and record metadata so retrieval quality can be checked before writing parquet.

## Error Handling

The command will produce clear errors for:

- a missing or empty corpus;
- malformed corpus JSONL;
- non-positive top-k;
- a query with no non-zero TF-IDF match;
- failure to import the parquet/dataset dependency stack.

Errors will name the relevant path or question and suggest the corrective action where useful.

## Documentation

The README dataset-preparation section will lead with the local smoke-test command and explain that it is offline and uses the 30-document demo corpus. The existing NQ command will move under an explicitly optional heading that lists its required Wikipedia corpus and precomputed retrieval-cache files. This makes it clear that running Search-QA preparation does not create those RAG prerequisites.

## Testing and Verification

Unit tests will verify:

- retrieval results are adapted into valid RAG records;
- a known corpus-backed question includes the expected document context;
- missing retrieval results fail clearly;
- preview output exposes the key audit fields;
- argument validation rejects invalid top-k values.

Integration verification will run the preview command and write a temporary parquet output, then inspect its row count and schema. The existing focused data and demo-retriever tests will also run to guard the reused boundaries.

## Success Criteria

A fresh checkout with project dependencies and the existing demo corpus can run one documented command without network access, inspect sensible retrieved context, and create a valid compact RAG parquet dataset. The README distinguishes this smoke test from full NQ preparation and accurately states the latter's external inputs.
