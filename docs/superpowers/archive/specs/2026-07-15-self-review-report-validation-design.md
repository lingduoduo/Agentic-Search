# Self-Review Report Validation Design

## Purpose

Make the existing subagent-driven development self-review handoff enforceable
without committing scratch reports or adding a CI workflow. Before an independent
task reviewer receives a report, a local validator must confirm that the report
contains the information required to review the implementation responsibly.

Validation establishes structural completeness only. It does not establish that
tests ran, claims are true, or the implementation is correct. The independent
diff-based reviewer remains responsible for technical judgment and must continue
to treat the report as unverified implementer claims.

## Scope

This change covers newly produced `.superpowers/sdd/*report.md` files and the
Superpowers subagent-driven development workflow that creates and consumes them.
Reports remain ignored local scratch artifacts. No reports, review manifests, or
CI jobs become tracked repository artifacts.

Existing reports are not migrated in bulk. An in-progress report must be brought
into compliance before its next independent task review. Validation is based on
report content, so descriptive filenames such as `structured-task-2-report.md`
remain supported.

## Canonical Report Contract

A task report must contain these level-two Markdown sections exactly once:

1. `Status`
2. `Commits`
3. `Implementation`
4. `Files changed`
5. `Test evidence`
6. `Self-review`
7. `Concerns`

Section order is fixed to keep reports predictable for implementers and reviewers.
Each section must contain substantive, non-placeholder content. `Concerns` may
contain the exact value `None`; all other sections require task-specific content.

`Status` must contain exactly one of the established workflow states:

- `DONE`
- `DONE_WITH_CONCERNS`
- `BLOCKED`
- `NEEDS_CONTEXT`

`Commits` must contain one or more bullet items whose first inline-code value is
an abbreviated or full Git commit SHA, optionally followed by its subject. Every
listed SHA must resolve unambiguously in the current repository through read-only
Git resolution. Symbolic references such as `HEAD` do not satisfy the contract.

`Test evidence` must contain at least one level-three evidence block. A normal
block is headed `### Test: NAME`, where `NAME` is descriptive. A TDD report uses the
required `### RED` and `### GREEN` blocks and may add normal test blocks for
regressions. Every evidence block contains exactly one `Command:` label followed
by a nonempty fenced code block and one `Result:` label followed by nonempty
prose. The validator checks this structure, not whether the command was executed
or whether the result is truthful. With `--require-tdd`, RED must also contain an
`Expected failure:` label with a nonempty explanation, and GREEN's result must
state a passing outcome.

Placeholder markers are rejected case-insensitively when they occur as standalone
markers or template tokens: `TODO`, `TBD`, `FIXME`, `PLACEHOLDER`, and unexpanded
angle-bracket or square-bracket template fields. Ordinary prose and Markdown links
must not be mistaken for placeholders.

## Validator Interface

Add a standard-library-only command:

```text
python scripts/validate_task_report.py [--require-tdd] REPORT_FILE
```

The command parses Markdown headings directly; it does not introduce a Markdown
parser dependency. It validates the entire report and prints all detected
violations in document order so an implementer can correct them in one pass.

Exit codes are:

- `0`: report satisfies the contract;
- `1`: one or more contract violations;
- `2`: invalid invocation, unreadable input, or inability to inspect the Git
  repository.

Diagnostics identify the report and affected section or line where possible. A
successful run prints a concise confirmation naming the validated report.

The validator never writes the report, executes reported test commands, changes
Git state, or contacts a remote service.

## Workflow Integration

The implementer prompt will require the canonical headings and instruct the
implementer to run the validator after writing the report. It will retain the
existing short final response containing status, commits, test summary, concerns,
and report path.

The subagent-driven development controller workflow will make validation a hard
pre-review gate:

1. Receive the implementer's short status response.
2. Run the validator against the report, adding `--require-tdd` when the task
   required TDD.
3. If validation fails, return the diagnostics to the implementer and require a
   report-only correction. Do not generate a review package or dispatch a reviewer.
4. When validation passes, generate the review package and dispatch the independent
   reviewer with the brief, validated report, diff package, and global constraints.

Fix agents append their evidence to the appropriate canonical sections instead of
creating arbitrary new top-level formats. Before re-review, the controller runs
the validator again so amended test evidence and concerns remain complete.

The progress ledger is updated only after a validated report and a clean independent
review, preserving its role as the session-recovery map.

## Components

### Report parser

The parser reads UTF-8 text, records Markdown heading level, title, body, and line
number, and exposes the seven required sections plus any `RED` and `GREEN`
subsections. Fenced code blocks are ignored while detecting headings so commands
inside examples cannot create false sections.

### Contract validator

Pure validation functions check required sections, uniqueness, order, substantive
content, status, placeholders, test evidence, and optional TDD evidence. They
return structured diagnostics rather than printing or exiting, allowing focused
unit tests.

### Git commit resolver

A narrow adapter invokes `git rev-parse --verify SHA^{commit}` without a shell.
It distinguishes an invalid commit from an unavailable repository. The adapter is
injected or isolated so tests can exercise validation deterministically.

### CLI

The CLI owns argument parsing, file and repository errors, diagnostic rendering,
and exit codes. It contains no report-contract logic.

## Error Handling

Malformed Markdown is reported as contract violations when it can be parsed safely.
Duplicate or out-of-order required sections are reported independently. Missing
sections do not prevent validation of sections that are present.

Unreadable files, invalid UTF-8, and Git inspection failures are operational errors
with exit code `2`, because the validator cannot determine report validity. Invalid
or ambiguous commit SHAs are report violations with exit code `1`.

The validator reports every known issue in one invocation and never silently
normalizes headings or content. This prevents an apparently successful validation
from masking a report that reviewers will interpret differently.

## Testing

Focused unit and CLI tests cover:

- a valid non-TDD report;
- a valid TDD report;
- each missing required section;
- duplicate and out-of-order sections;
- empty content and allowed `Concerns: None`;
- all valid and invalid status values;
- standalone placeholders without false positives for normal Markdown links;
- missing command or result evidence;
- missing, empty, or semantically incomplete RED/GREEN evidence;
- valid full and abbreviated commit SHAs;
- invalid and ambiguous commit references;
- multiple diagnostics returned in one run;
- unreadable files, invalid UTF-8, and non-repository execution;
- fenced headings ignored by the parser;
- exact CLI exit-code behavior.

Git-related tests use temporary repositories and local commits. Tests do not depend
on the developer's current branch, global Git configuration, network, or remote
services.

## Acceptance Criteria

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
- The implementer and controller workflow documentation makes successful
  validation mandatory before initial review and re-review.
- Reports remain ignored local artifacts and no CI workflow is added.
