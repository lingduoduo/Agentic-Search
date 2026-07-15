# Generated Context Pack

# Self Review Report Validation

## Sources

- [Specification: 2026-07-15-self-review-report-validation-design.md](../archive/specs/2026-07-15-self-review-report-validation-design.md)
- [Plan: 2026-07-15-self-review-report-validation.md](../archive/plans/2026-07-15-self-review-report-validation.md)

## Specification Context

### Purpose

Make the existing subagent-driven development self-review handoff enforceable
without committing scratch reports or adding a CI workflow. Before an independent
task reviewer receives a report, a local validator must confirm that the report
contains the information required to review the implementation responsibly.

Validation establishes structural completeness only. It does not establish that
tests ran, claims are true, or the implementation is correct. The independent
diff-based reviewer remains responsible for technical judgment and must continue
to treat the report as unverified implementer claims.

### Scope

This change covers newly produced `.superpowers/sdd/*report.md` files and the
Superpowers subagent-driven development workflow that creates and consumes them.
Reports remain ignored local scratch artifacts. No reports, review manifests, or
CI jobs become tracked repository artifacts.

Existing reports are not migrated in bulk. An in-progress report must be brought
into compliance before its next independent task review. Validation is based on
report content, so descriptive filenames such as `structured-task-2-report.md`
remain supported.

### Acceptance Criteria

- A canonical task report passes the validator and returns exit code `0`.
- Any missing, duplicated, out-of-order, empty, or template-marker required section
  causes exit code `1` with an actionable diagnostic.
- Only the four established statuses are accepted.
- At least one resolvable commit SHA is required.
- At least one precisely formatted evidence block is required; `--require-tdd`
  additionally enforces RED and GREEN blocks with an explained expected failure
  and a stated passing result.
- All detectable violations are reported in one run.
- Operational failures return exit code `2` without changing files or Git state.

…

## Implementation Plan Context

### Task 1: Markdown Parser and Pure Contract Validation

**Files:**
- Create: `scripts/validate_task_report.py`
- Create: `tests/unit/test_validate_task_report.py`

**Interfaces:**
- Produces: `Diagnostic(line: int, section: str | None, message: str)`.
- Produces: `Section(level: int, title: str, body: str, line: int)`.
- Produces: `ParsedReport(sections: tuple[Section, ...])`.
- Produces: `parse_report(text: str) -> ParsedReport`.
- Produces: `validate_report(report: ParsedReport, *, resolve_commit: Callable[[str], bool], require_tdd: bool = False) -> tuple[Diagnostic, ...]`.
- Does not perform file I/O, print, exit, or invoke Git from pure parsing and validation functions.

…

### Task 2: Git-Aware CLI and Operational Exit Codes

**Files:**
- Modify: `scripts/validate_task_report.py`
- Modify: `tests/unit/test_validate_task_report.py`

**Interfaces:**
- Consumes: `parse_report` and `validate_report` from Task 1.
- Produces: `GitInspectionError(RuntimeError)`.
- Produces: `resolve_git_commit(sha: str, *, cwd: Path) -> bool`.
- Produces: `main(argv: Sequence[str] | None = None) -> int`.
- CLI: `python scripts/validate_task_report.py [--require-tdd] REPORT_FILE`.

- [ ] **Step 1: Add temporary-repository and CLI tests**

Add helpers that create isolated repositories without global identity dependencies:

…

### Task 3: Repository Workflow Enforcement and Documentation

**Files:**
- Create: `AGENTS.md`
- Create: `docs/development/self-review-reports.md`
- Modify: `README.md:138` (Documentation section)
- Move: `docs/superpowers/specs/2026-07-15-self-review-report-validation-design.md` to `docs/superpowers/archive/specs/2026-07-15-self-review-report-validation-design.md`
- Move: `docs/superpowers/plans/2026-07-15-self-review-report-validation.md` to `docs/superpowers/archive/plans/2026-07-15-self-review-report-validation.md`
- Generate: `docs/superpowers/context-packs/self-review-report-validation-context-pack.md`
- Modify generated: `docs/superpowers/context-packs/INDEX.md`
- Test: `tests/unit/test_validate_task_report.py`

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
