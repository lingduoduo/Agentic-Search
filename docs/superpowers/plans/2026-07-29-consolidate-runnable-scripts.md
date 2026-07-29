# Consolidate Runnable Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish relocating `scripts/generate_context_packs.py` and `scripts/validate_task_report.py` into `examples/` so every live reference points at the new home and the repository has one directory for runnable Python entry points.

**Architecture:** The two scripts have already been copied to `examples/` byte-identically and deleted from `scripts/` in the working tree, but no reference was repointed, so two test files fail at import. This plan stages the moves as git renames and repoints imports, documentation, and one test assertion. No script contents change.

**Tech Stack:** Python 3.10+, pytest, ruff, git.

Spec: `docs/superpowers/specs/2026-07-29-consolidate-runnable-scripts-design.md`

## Global Constraints

- Branch is `refactor/consolidate-runnable-scripts`. Never commit to `main`.
- The contents of `examples/generate_context_packs.py` and `examples/validate_task_report.py` must not change. They are byte-identical to their former `scripts/` versions and must stay that way, so git records pure renames.
- Do not edit anything under `docs/superpowers/archive/`, `docs/superpowers/specs/`, or `docs/superpowers/context-packs/`. Those are records of past work.
- Do not touch `src/internal/`, `bin/`, `cli/`, `src/cli/`, or root-level `colab-vllm.py` beyond the two file deletions named in Task 3.
- `AGENTS.md` and the assertion in `tests/unit/test_validate_task_report.py` are coupled: the test reads `AGENTS.md` and greps for a literal command string. They must change in the same commit.
- Run `ruff check . && ruff format --check .` before the final commit.

---

### Task 1: Repoint `validate_task_report` references

The relocation of this script touches four places at once: the test's import, a
fixture string inside the test, `AGENTS.md`, and the developer doc. The test
asserts on `AGENTS.md` content, so `AGENTS.md` cannot be split into a later
task.

**Files:**
- Rename (stage only, no content change): `scripts/validate_task_report.py` → `examples/validate_task_report.py`
- Modify: `tests/unit/test_validate_task_report.py:8`, `:34`, `:530`
- Modify: `AGENTS.md:8`
- Modify: `docs/development/self-review-reports.md:117-118`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: the module path `examples.validate_task_report`, exporting `GitInspectionError`, `main`, `parse_report`, `resolve_git_commit`, `validate_report` — unchanged names and signatures, only the package prefix moves from `scripts` to `examples`.

- [ ] **Step 1: Run the test to observe the RED state**

Run: `pytest tests/unit/test_validate_task_report.py -v`

Expected: collection error, `ModuleNotFoundError: No module named 'scripts'`. This is the failing state the task fixes — do not write a new test, this existing suite is the regression check.

- [ ] **Step 2: Repoint the test import**

In `tests/unit/test_validate_task_report.py`, change line 8 from:

```python
from scripts.validate_task_report import (
```

to:

```python
from examples.validate_task_report import (
```

Leave the imported names and the rest of the import block untouched.

- [ ] **Step 3: Update the fixture path string**

In the same file, inside `canonical_report()`, line 34 reads:

```python
- `scripts/validate_task_report.py`
```

Change it to:

```python
- `examples/validate_task_report.py`
```

This is fixture text under a "Files changed" heading. The validator only checks that the heading exists, not that the path resolves, so this edit is for consistency and cannot break the assertion.

- [ ] **Step 4: Update `AGENTS.md` and the assertion that reads it**

In `AGENTS.md`, line 8 reads:

```
Run `python scripts/validate_task_report.py REPORT_FILE` after the implementer
```

Change it to:

```
Run `python examples/validate_task_report.py REPORT_FILE` after the implementer
```

In `tests/unit/test_validate_task_report.py`, line 530 reads:

```python
    assert "python scripts/validate_task_report.py" in instructions
```

Change it to:

```python
    assert "python examples/validate_task_report.py" in instructions
```

Both edits are required together: `test_repository_agent_instructions_require_report_validation` reads `AGENTS.md` from disk and greps for that literal string.

- [ ] **Step 5: Update the developer documentation**

In `docs/development/self-review-reports.md`, lines 117-118 read:

```bash
python scripts/validate_task_report.py REPORT_FILE
python scripts/validate_task_report.py --require-tdd REPORT_FILE
```

Change both to:

```bash
python examples/validate_task_report.py REPORT_FILE
python examples/validate_task_report.py --require-tdd REPORT_FILE
```

- [ ] **Step 6: Run the test to verify GREEN**

Run: `pytest tests/unit/test_validate_task_report.py -v`

Expected: all tests pass, including `test_repository_agent_instructions_require_report_validation`.

- [ ] **Step 7: Confirm git records a rename, not a delete plus add**

Run:

```bash
git add examples/validate_task_report.py scripts/validate_task_report.py
git status --short
```

Expected: a single line beginning with `R` mapping `scripts/validate_task_report.py -> examples/validate_task_report.py`. If it shows `D` and `A` instead, the file contents diverged — restore with `git show HEAD:scripts/validate_task_report.py > examples/validate_task_report.py` and re-stage.

- [ ] **Step 8: Commit**

```bash
git add examples/validate_task_report.py scripts/validate_task_report.py \
        tests/unit/test_validate_task_report.py AGENTS.md \
        docs/development/self-review-reports.md
git commit -m "refactor: move validate_task_report into examples/"
```

---

### Task 2: Repoint `generate_context_packs` references

**Files:**
- Rename (stage only, no content change): `scripts/generate_context_packs.py` → `examples/generate_context_packs.py`
- Modify: `tests/unit/test_generate_context_packs.py:5`

**Interfaces:**
- Consumes: nothing from Task 1. The two scripts are independent.
- Produces: the module path `examples.generate_context_packs`, exporting `discover_sources`, `generate`, `normalize_source`, `pair_sources`, `validate_generated` — unchanged names and signatures.

- [ ] **Step 1: Run the test to observe the RED state**

Run: `pytest tests/unit/test_generate_context_packs.py -v`

Expected: collection error, `ModuleNotFoundError: No module named 'scripts'`.

- [ ] **Step 2: Repoint the test import**

In `tests/unit/test_generate_context_packs.py`, change line 5 from:

```python
from scripts.generate_context_packs import (
```

to:

```python
from examples.generate_context_packs import (
```

Leave the imported names untouched.

- [ ] **Step 3: Run the test to verify GREEN**

Run: `pytest tests/unit/test_generate_context_packs.py -v`

Expected: all tests pass.

- [ ] **Step 4: Confirm git records a rename**

Run:

```bash
git add examples/generate_context_packs.py scripts/generate_context_packs.py
git status --short
```

Expected: a single line beginning with `R` mapping `scripts/generate_context_packs.py -> examples/generate_context_packs.py`.

- [ ] **Step 5: Commit**

```bash
git add examples/generate_context_packs.py scripts/generate_context_packs.py \
        tests/unit/test_generate_context_packs.py
git commit -m "refactor: move generate_context_packs into examples/"
```

---

### Task 3: Remove `scripts/` and the stray vendored files

**Files:**
- Modify: `examples/beir_to_corpus.py:13-14`
- Delete (untracked, filesystem only): `src/internal/document_index/base.py`, `src/internal/document_index/document_processing.py`
- Delete (untracked, filesystem only): the `scripts/` directory

**Interfaces:**
- Consumes: Tasks 1 and 2 must be committed first, so `scripts/` holds no tracked files when it is removed.
- Produces: nothing later tasks depend on. This is the final task.

- [ ] **Step 1: Fix the stale docstring in `beir_to_corpus.py`**

`examples/beir_to_corpus.py` lines 13-14 still document the script under its old
path — leftover from an earlier relocation with the same gap. They read:

```
    python3 scripts/beir_to_corpus.py --dataset nfcorpus
    python3 scripts/beir_to_corpus.py --dataset scifact --out data/corpus_scifact.jsonl
```

Change to:

```
    python3 examples/beir_to_corpus.py --dataset nfcorpus
    python3 examples/beir_to_corpus.py --dataset scifact --out data/corpus_scifact.jsonl
```

- [ ] **Step 2: Verify `scripts/` holds nothing tracked, then remove it**

Run:

```bash
git ls-files scripts/
```

Expected: empty output. If it prints any path, stop — Task 1 or Task 2 did not commit its rename.

Then remove the directory and its untracked leftovers (`.DS_Store`, `__pycache__`):

```bash
rm -rf scripts/
```

- [ ] **Step 3: Delete the stray vendored files**

These are untracked AWorld reference code, unimported anywhere, superseded by
`src/internal/mcp_server/tools/documents.py` and
`src/internal/mcp_server/document_parser_runtime.py` from PR #472.

First confirm nothing imports them:

```bash
grep -rn "document_index.base\|document_index import base\|document_processing" --include="*.py" src tests examples
```

Expected: no hits outside the two files themselves. Then:

```bash
rm src/internal/document_index/base.py src/internal/document_index/document_processing.py
```

- [ ] **Step 4: Verify no live `scripts/` references remain**

Run:

```bash
grep -rn "scripts/" --include="*.py" --include="*.md" --include="*.toml" --include="*.sh" . \
  | grep -v node_modules | grep -v "docs/superpowers/"
```

Expected: no output. Hits under `docs/superpowers/` are historical records and are intentionally left alone.

- [ ] **Step 5: Run the full suite and lint**

```bash
pytest
ruff check . && ruff format --check .
```

Expected: the suite passes and lint is clean. If `ruff format --check` reports a diff, run `ruff format .` and re-run. Note that the repository's pre-commit hook runs ruff-format and will abort a commit if it reformats files — run the formatter before committing rather than during.

- [ ] **Step 6: Commit**

```bash
git add examples/beir_to_corpus.py
git commit -m "docs: point beir_to_corpus usage at examples/"
```

The `scripts/` removal and the two stray deletions produce no diff — every file involved was untracked — so they need no commit of their own.

- [ ] **Step 7: Push and open the pull request**

```bash
git push -u origin refactor/consolidate-runnable-scripts
gh pr create --title "refactor: consolidate runnable scripts into examples/" --body "..."
```

The PR body should state that the two scripts moved byte-identically, list the repointed references, and note that historical `docs/superpowers/` references were deliberately left as written.
