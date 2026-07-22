# Plan: Data-source cards must never render "Unknown"

Spec: `docs/superpowers/specs/2026-07-22-source-label-not-unknown.md`

## Step 1 — Backend: label the classic RAG path
`src/internal/servers/web/app.py`, `_run_agent_impl` fallback branch.
Wrap `result.context.documents` with `_document_with_metadata(...,
source_provider="retrieval", query=query, entry_point="rag")` before passing
them to `_finalize_response`.
→ verify: new unit test asserts `metadata["source"] == "Local Retrieval"`.

## Step 2 — Frontend: drop the "Unknown" fallback
`web/src/components/SourceGrid.tsx`. Default a missing/non-string
`metadata.source` to "Local Retrieval" instead of "Unknown".
→ verify: `cd web && npm run typecheck`; grep shows no "Unknown" literal.

## Step 3 — Regression test
`tests/unit/servers/web/test_agent_dispatch.py` (or nearest existing dispatch
test): drive the classic path with a stubbed `answer_with_retrieval` and assert
the finalized documents carry `source == "Local Retrieval"`.
→ verify: `pytest -k source_label`.

## Step 4 — Corpus/index refresh
The demo/hybrid retrieval servers rebuild their in-memory index from
`data/corpus.jsonl` on every startup — there is no persisted local index to
drop. Confirm this and report; build a fresh FAISS/BM25 index via the
document-index CLI only if a persisted index is actually in use.
→ verify: document what was (or was not) dropped/rebuilt in the PR body.

## Step 5 — Ship
`pytest`, `npm run typecheck`, `ruff check . --fix && ruff format .`, commit,
push, open PR.
