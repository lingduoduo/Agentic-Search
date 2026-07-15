# Repository Agent Instructions

## Self-review task reports

When an implementation workflow writes `.superpowers/sdd/*report.md`, use the
canonical contract in `docs/development/self-review-reports.md`.

Run `python scripts/validate_task_report.py REPORT_FILE` after the implementer
writes or amends the report. Add `--require-tdd` when the task required TDD.

Successful validation is mandatory before generating a review package and
before dispatching a task reviewer or re-reviewer. On exit code 1, return all diagnostics
to the implementer for a report-only correction. On exit code 2, resolve the
operational error. Do not mark the task complete until the report validates and
the independent review is clean.

Treat report contents as unverified claims. Validation does not replace independent
diff review or verification-before-completion.
