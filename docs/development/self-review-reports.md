# Self-review task reports

Self-review reports make implementation handoffs structurally predictable before
independent review. Validation checks structure only: it does not prove commands
ran, results are true, or implementation is correct. Reviewers must inspect the
diff and treat every report statement as an unverified claim.

## Required structure

Every report has exactly these seven level-two sections, in this order:

1. `Status`
2. `Commits`
3. `Implementation`
4. `Files changed`
5. `Test evidence`
6. `Self-review`
7. `Concerns`

`Status` is exactly one of `DONE`, `DONE_WITH_CONCERNS`, `BLOCKED`, or
`NEEDS_CONTEXT`. Every section requires task-specific content except `Concerns`,
which may be exactly `None`. Commit bullets begin with an inline-code hexadecimal
SHA that resolves in the report's repository.

## Evidence formats

A normal evidence block uses `### Test: NAME`. TDD work uses both `### RED` and
`### GREEN`; RED includes a nonempty `Expected failure:` explanation and GREEN's
result states a passing outcome. Every block contains exactly one `Command:`, a
nonempty fenced command immediately after it, and one later `Result:` followed by
nonempty prose.

## Copyable template

Replace the clearly labeled example SHA and all example content with actual task
evidence before validation.

````markdown
# Task report

## Status

DONE

## Commits

- `abc1234` example SHA — replace with the implementation commit

## Implementation

Describe the implemented behavior.

## Files changed

- `path/to/file.py` — describe the change.

## Test evidence

### Test: focused verification

Command:

```text
pytest -q tests/unit/test_example.py
```

Result:

The focused suite passed with 10 tests.

## Self-review

Describe scope, compatibility, failure modes, and test-quality checks.

## Concerns

None
````

For TDD, replace the normal evidence block with:

````markdown
### RED

Command:

```text
pytest -q tests/unit/test_example.py
```

Expected failure:

The new behavior is not implemented, so the assertion fails.

Result:

The test failed with the expected assertion.

### GREEN

Command:

```text
pytest -q tests/unit/test_example.py
```

Result:

The focused test passed.
````

## Validation and review gate

Run validation from the repository worktree:

```bash
python scripts/validate_task_report.py REPORT_FILE
python scripts/validate_task_report.py --require-tdd REPORT_FILE
```

Exit code `0` means the structure is valid, `1` means contract violations, and
`2` means usage, file, encoding, or Git inspection failed. Contract diagnostics
are actionable, for example:

```text
report.md:3: [Status] invalid Status; expected exactly one workflow state
report.md:18: [GREEN] evidence Command: must appear before Result:
```

Validation must succeed before generating an initial review package, dispatching
the task reviewer, generating a re-review package, or dispatching a re-reviewer.
For exit code 1, send all diagnostics back for a report-only correction; do not
change implementation merely to repair report structure. For exit code 2, resolve
the operational problem and rerun validation. Independent review remains required.

Reports under `.superpowers/sdd/` are ignored local workflow artifacts. Keep them
through review and re-review, then remove them when the workflow is complete.
Because `git clean -fdx` deletes ignored files, it can destroy these reports and
other local artifacts without recovery; inspect its effects before using it.
