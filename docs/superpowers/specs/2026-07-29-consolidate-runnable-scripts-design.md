# Consolidate runnable scripts into `examples/`

**Date:** 2026-07-29
**Status:** Approved

## Problem

The repository had two homes for runnable Python entry points: `scripts/` and
`examples/`. A partial relocation left the tree in a broken intermediate state:

- `scripts/generate_context_packs.py` and `scripts/validate_task_report.py` were
  copied to `examples/` (byte-identical) and deleted from `scripts/`, but no
  reference was repointed.
- `tests/unit/test_validate_task_report.py` and
  `tests/unit/test_generate_context_packs.py` still import `from scripts.…`, so
  both files fail at collection.
- `AGENTS.md` and `docs/development/self-review-reports.md` still document the
  `scripts/` invocation. `test_validate_task_report.py` asserts on the literal
  `AGENTS.md` string, so the doc and the test are coupled.
- `scripts/` now holds no tracked files.

Separately, two untracked files were dropped into `src/internal/document_index/`:
`base.py` ("Base models and utilities for perception tools MCP server") and
`document_processing.py` ("Based on AWorld MCP server implementation"). They are
vendored reference code — nothing imports them, they belong to no package
boundary in `document_index/` (which owns chunking, embedding, and FAISS), and
the hardened equivalent already shipped in PR #472 as
`src/internal/mcp_server/tools/documents.py` plus
`src/internal/mcp_server/document_parser_runtime.py`.

## Goal

One home for runnable entry points (`examples/`), with every live reference
pointing at it, and the stray vendored files removed.

## Design

The two moved scripts keep their contents unchanged. All other edits repoint
references.

### Changes

1. Stage `scripts/generate_context_packs.py` and `scripts/validate_task_report.py`
   as git renames into `examples/`. Content is unchanged, so git records pure
   renames.
2. `tests/unit/test_validate_task_report.py` — import `from
   examples.validate_task_report`; update the `scripts/validate_task_report.py`
   reference in the module docstring; update the `AGENTS.md` assertion to expect
   `python examples/validate_task_report.py`.
3. `tests/unit/test_generate_context_packs.py` — import `from
   examples.generate_context_packs`.
4. `AGENTS.md` — the report-validation mandate invokes
   `python examples/validate_task_report.py`.
5. `docs/development/self-review-reports.md` — both CLI invocations use
   `examples/`.
6. `examples/beir_to_corpus.py` — module docstring documents itself as
   `examples/beir_to_corpus.py`, not `scripts/`. Leftover from an earlier
   relocation with the same gap.
7. Remove the `scripts/` directory. Its only remaining entries are untracked
   (`.DS_Store`, `__pycache__`), so this produces no repo delta beyond the
   renames.
8. Delete `src/internal/document_index/base.py` and
   `src/internal/document_index/document_processing.py`. Both are untracked, so
   this is a filesystem change with no diff.

`examples/__init__.py` already exists, so `from examples.<module> import …`
resolves the same way `from scripts.<module> import …` did — via the repository
root on `sys.path`. Neither moved script derives paths from `__file__`, so
relocation does not change their runtime behavior.

### Out of scope

References under `docs/superpowers/archive/`, `docs/superpowers/specs/`, and
`docs/superpowers/context-packs/` are records of past work and stay as
written.

`docs/superpowers/plans/` is treated differently. It is a mixed directory —
archiving to `archive/plans/` lapsed, so it now holds both unstarted plans and
plans whose work has already merged. Three of them
(`2026-07-29-mcp-document-extraction.md`, `2026-07-15-tool-result-size-bound.md`,
`2026-07-15-llm-timeout-degraded-answer.md`) instruct an agent to run the
validator, so they were repointed to `examples/validate_task_report.py`. The
reason is not that they are known to be live: it is that an instruction naming a
deleted path is useless either way, and repointing it costs one line. The rest of
the directory is left alone.

No consolidation of `src/internal/` packages, `bin/`, `cli/`, `src/cli/`, or
root-level `colab-vllm.py`. Those are separate questions.

## Verification

The two affected test files are the regression check. They fail at import today
(RED) and pass after the repointing (GREEN):

```bash
pytest tests/unit/test_validate_task_report.py tests/unit/test_generate_context_packs.py -v
```

Then the full suite and lint:

```bash
pytest
ruff check . && ruff format --check .
```

A grep for live `scripts/` references outside `docs/superpowers/` must come back
empty.

## Risks

Low. No behavior changes, no production code paths touched. The one coupling to
watch is `AGENTS.md` and the assertion in `test_validate_task_report.py`, which
must change together or the test fails.
