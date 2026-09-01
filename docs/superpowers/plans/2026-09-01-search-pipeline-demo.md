# Plan: rewrite the search-pipeline example over the real pipeline

Spec: `docs/superpowers/specs/2026-09-01-search-pipeline-demo-design.md`

## Task 1 — rewrite the tests against the real types

1. Replace the two `run_search_pipeline` tests in
   `tests/unit/test_readme_examples.py` with tests over the new interface:
   anonymous / group / email callers, the `--skip-enforcement` leak, and
   adjacent-chunk merging.
   → verify: tests fail (the new names do not exist yet).
2. Leave `test_grpo_training_pipeline_example_runs_without_model_backends`
   untouched.

## Task 2 — rewrite the example

1. Replace `examples/run_search_pipeline.py` with the four-step script over
   `SearchFilters`, `build_user_only_filters`, `TfidfRetriever.from_docs`,
   `documents_from_search_results` and `merge_adjacent_documents`.
2. Add the `--skip-enforcement` flag and the per-step id printing.
   → verify: new tests pass; `python3 -m examples.run_search_pipeline` prints a
   filtered result set and `--skip-enforcement` prints a larger one.

## Task 3 — docs

1. `docs/training-and-evaluation.md` describes the example as a filter/ACL
   walkthrough; update it to say what the script now demonstrates and mention
   `--skip-enforcement`.
   → verify: the command in the doc runs as written.

## Task 4 — verification

1. `pytest` (full default suite).
2. `python3 -m examples.run_search_pipeline` and `--skip-enforcement`, output
   compared.
3. `ruff check . --fix && ruff format .`
4. Mutation-check the enforcement test: drop the `filters.matches` call and
   confirm the leak test goes red. Clear stale `.pyc` before re-running.
