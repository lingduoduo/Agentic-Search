# Self-Review Report Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local, mandatory pre-review validator for task self-review reports and enforce its use through repository agent instructions.

**Architecture:** A focused standard-library Python module will parse Markdown reports, validate the canonical contract, and resolve commit SHAs through a narrow injected adapter. Its CLI will own operational errors and exit codes. A tracked root `AGENTS.md` and a concise workflow document will make successful validation mandatory before review-package generation and reviewer dispatch.

**Tech Stack:** Python 3.12 standard library, `argparse`, `dataclasses`, `pathlib`, `re`, `subprocess`, pytest, Ruff, Markdown documentation.

## Global Constraints

- Reports remain ignored local scratch artifacts under `.superpowers/sdd/`; do not commit reports or add a CI workflow.
- The validator must never write reports, execute reported commands, mutate Git state, or contact a remote service.
- Validation establishes structural completeness only; independent reviewers must continue treating report claims as unverified.
- Required level-two sections, exactly once and in order: `Status`, `Commits`, `Implementation`, `Files changed`, `Test evidence`, `Self-review`, `Concerns`.
- Accepted statuses are exactly `DONE`, `DONE_WITH_CONCERNS`, `BLOCKED`, and `NEEDS_CONTEXT`.
- `Concerns` may contain the exact value `None`; every other required section must contain substantive non-placeholder content.
- Commit entries are bullet items whose first inline-code value is an abbreviated or full hexadecimal SHA; every SHA must resolve unambiguously as a commit.
- A normal evidence block is `### Test: NAME`; TDD evidence uses `### RED` and `### GREEN`.
- Every evidence block contains exactly one `Command:` label followed by a nonempty fenced code block and one `Result:` label followed by nonempty prose.
- With `--require-tdd`, RED additionally contains a nonempty `Expected failure:` explanation and GREEN states a passing result.
- Exit codes are exactly `0` for valid, `1` for contract violations, and `2` for invocation, file, encoding, or Git-inspection failures.
- Report all detectable contract violations in one invocation in document order.
- No third-party Markdown or validation dependency may be added.

---

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

- [ ] **Step 1: Write parser tests that establish heading and fence behavior**

Add helpers and tests covering the canonical section order and fenced fake headings:

```python
from __future__ import annotations

from scripts.validate_task_report import parse_report


def canonical_report(*, status: str = "DONE", commit: str = "abc1234") -> str:
    return f"""# Task 1 Report

## Status

{status}

## Commits

- `{commit}` implementation commit

## Implementation

Implemented the requested validator behavior.

## Files changed

- `scripts/validate_task_report.py`

## Test evidence

### Test: focused unit suite

Command:

```text
pytest -q tests/unit/test_validate_task_report.py
```

Result:

The focused suite passed with 20 tests.

## Self-review

Checked completeness, scope, naming, edge cases, and test quality.

## Concerns

None
"""


def test_parse_report_ignores_headings_inside_fences() -> None:
    text = canonical_report().replace(
        "pytest -q tests/unit/test_validate_task_report.py",
        "## Status\npytest -q tests/unit/test_validate_task_report.py",
    )

    report = parse_report(text)

    assert [section.title for section in report.sections if section.level == 2] == [
        "Status",
        "Commits",
        "Implementation",
        "Files changed",
        "Test evidence",
        "Self-review",
        "Concerns",
    ]
```

- [ ] **Step 2: Run the parser test to verify RED**

Run: `pytest -q tests/unit/test_validate_task_report.py::test_parse_report_ignores_headings_inside_fences`

Expected: collection fails with `ModuleNotFoundError` or `ImportError` because `scripts.validate_task_report` does not exist.

- [ ] **Step 3: Implement fenced-heading parsing**

Create `scripts/validate_task_report.py` with frozen data objects and a parser that records heading lines and bodies while ignoring headings inside matching backtick or tilde fences:

```python
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")


@dataclass(frozen=True, order=True)
class Diagnostic:
    line: int
    section: str | None
    message: str


@dataclass(frozen=True)
class Section:
    level: int
    title: str
    body: str
    line: int


@dataclass(frozen=True)
class ParsedReport:
    sections: tuple[Section, ...]


def parse_report(text: str) -> ParsedReport:
    headings: list[tuple[int, str, int, int]] = []
    fence: tuple[str, int] | None = None
    lines = text.splitlines()
    for index, line in enumerate(lines, start=1):
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = (marker[0], len(marker))
            elif marker[0] == fence[0] and len(marker) >= fence[1]:
                fence = None
            continue
        if fence is not None:
            continue
        heading_match = HEADING_RE.match(line)
        if heading_match:
            headings.append(
                (len(heading_match.group(1)), heading_match.group(2).strip(), index, index)
            )
    sections: list[Section] = []
    for offset, (level, title, line, body_start) in enumerate(headings):
        body_end = headings[offset + 1][2] - 1 if offset + 1 < len(headings) else len(lines)
        body = "\n".join(lines[body_start:body_end]).strip()
        sections.append(Section(level=level, title=title, body=body, line=line))
    return ParsedReport(tuple(sections))
```

- [ ] **Step 4: Add pure validation tests**

Add parametrized or individually named tests for:

```python
from scripts.validate_task_report import validate_report


def messages(text: str, *, require_tdd: bool = False) -> list[str]:
    diagnostics = validate_report(
        parse_report(text),
        resolve_commit=lambda sha: sha == "abc1234",
        require_tdd=require_tdd,
    )
    return [diagnostic.message for diagnostic in diagnostics]


def test_valid_non_tdd_report_has_no_diagnostics() -> None:
    assert messages(canonical_report()) == []


def test_reports_all_missing_required_sections() -> None:
    assert messages("# Task report\n") == [
        "missing required section: Status",
        "missing required section: Commits",
        "missing required section: Implementation",
        "missing required section: Files changed",
        "missing required section: Test evidence",
        "missing required section: Self-review",
        "missing required section: Concerns",
    ]


def test_tdd_requires_red_green_and_expected_failure() -> None:
    errors = messages(canonical_report(), require_tdd=True)
    assert "missing TDD evidence block: RED" in errors
    assert "missing TDD evidence block: GREEN" in errors
```

Also cover every accepted and rejected status, duplicate sections, swapped sections, empty bodies, `Concerns` equal to `None`, standalone placeholder markers, unexpanded template fields, ordinary Markdown links, invalid commit bullet syntax, unresolved commits, missing command/result labels, duplicate labels, empty command fences, empty results, valid RED/GREEN, RED without `Expected failure:`, GREEN without a passing result, and aggregation of independent violations.

- [ ] **Step 5: Run the validation tests to verify RED**

Run: `pytest -q tests/unit/test_validate_task_report.py`

Expected: parser tests pass while validation tests fail because `validate_report` and contract checks are absent.

- [ ] **Step 6: Implement the pure contract validator**

Add these constants and validation boundaries:

```python
REQUIRED_SECTIONS = (
    "Status",
    "Commits",
    "Implementation",
    "Files changed",
    "Test evidence",
    "Self-review",
    "Concerns",
)
VALID_STATUSES = {"DONE", "DONE_WITH_CONCERNS", "BLOCKED", "NEEDS_CONTEXT"}
SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
COMMIT_ITEM_RE = re.compile(r"^[ \t]*-[ \t]+`([0-9a-fA-F]{7,40})`(?:[ \t]+.*)?$")
PLACEHOLDER_LINE_RE = re.compile(
    r"^[ \t]*(?:TODO|TBD|FIXME|PLACEHOLDER)[ \t]*$", re.IGNORECASE
)
TEMPLATE_TOKEN_RE = re.compile(r"<([A-Z][A-Z0-9_ -]*)>|\[([A-Z][A-Z0-9_ -]*)\]")
COMMAND_LABEL_RE = re.compile(r"(?m)^Command:[ \t]*$")
RESULT_LABEL_RE = re.compile(r"(?m)^Result:[ \t]*$")
EXPECTED_FAILURE_RE = re.compile(r"(?m)^Expected failure:[ \t]*(?:\n+)?(.+)$")
PASSING_RESULT_RE = re.compile(r"\b(?:pass|passed|passing|exit(?: code)?[ :=]+0)\b", re.IGNORECASE)
```

Implement small private functions for required-section lookup, substantive-body checks, status validation, commit extraction/resolution, evidence-block discovery, evidence label/fence checks, and optional TDD semantics. `validate_report` must call every applicable check, collect rather than short-circuit, and return diagnostics sorted by `(line, section or "", message)` while missing-section diagnostics retain required-section order.

Markdown links such as `[guide](docs/guide.md)` must be excluded before template-token scanning. Evidence blocks are the level-three sections between `Test evidence` and the next level-two section; only `RED`, `GREEN`, and titles beginning `Test: ` are accepted.

- [ ] **Step 7: Run focused tests and static checks**

Run: `pytest -q tests/unit/test_validate_task_report.py`

Expected: all Task 1 tests pass.

Run: `ruff check scripts/validate_task_report.py tests/unit/test_validate_task_report.py && ruff format --check scripts/validate_task_report.py tests/unit/test_validate_task_report.py && git diff --check`

Expected: all checks pass with no output from `git diff --check`.

- [ ] **Step 8: Commit Task 1**

```bash
git add scripts/validate_task_report.py tests/unit/test_validate_task_report.py
git commit -m "feat: validate self-review report contracts"
```

---

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

```python
import subprocess
from pathlib import Path

from scripts.validate_task_report import main, resolve_git_commit


def git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def make_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test User")
    git(repo, "config", "user.email", "test@example.com")
    (repo / "tracked.txt").write_text("content\n", encoding="utf-8")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-q", "-m", "test commit")
    return repo, git(repo, "rev-parse", "HEAD")
```

Test full and unique abbreviated SHAs, a nonexistent SHA returning `False`, execution outside a repository raising `GitInspectionError`, valid CLI output and exit `0`, aggregated violation output and exit `1`, missing file and invalid UTF-8 exit `2`, `--require-tdd`, and invalid invocation exit `2` through `argparse` behavior.

- [ ] **Step 2: Run CLI tests to verify RED**

Run: `pytest -q tests/unit/test_validate_task_report.py -k 'git or cli or main'`

Expected: failures or import errors for absent `GitInspectionError`, `resolve_git_commit`, and `main`.

- [ ] **Step 3: Implement Git resolution without a shell**

Add imports and implementation equivalent to:

```python
import subprocess
from pathlib import Path


class GitInspectionError(RuntimeError):
    pass


def resolve_git_commit(sha: str, *, cwd: Path) -> bool:
    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        raise GitInspectionError("unable to inspect commits outside a Git worktree")
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if resolved.returncode == 0:
        return True
    if resolved.returncode == 1:
        return False
    raise GitInspectionError(resolved.stderr.strip() or "Git commit inspection failed")
```

Catch `OSError` from process creation and translate it to `GitInspectionError` without exposing a traceback from the CLI.

- [ ] **Step 4: Implement CLI orchestration and exact diagnostics**

Use `argparse.ArgumentParser`, `Path.read_text(encoding="utf-8")`, and the report file's parent repository context. The CLI must render contract diagnostics as:

```text
REPORT:LINE: [SECTION] MESSAGE
```

Use `[report]` when no section exists. Success output is:

```text
Validated self-review report: REPORT
```

Operational errors go to stderr as `REPORT: ERROR` and return `2`. Contract errors go to stdout and return `1`. End the module with:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

The `resolve_commit` closure passed to `validate_report` must call
`resolve_git_commit(sha, cwd=report_path.parent)`; Git discovers the enclosing
worktree by walking parents.

- [ ] **Step 5: Run Task 2 tests and exercise the real CLI**

Run: `pytest -q tests/unit/test_validate_task_report.py`

Expected: all validator tests pass.

Create a canonical scratch report using the current commit SHA and run:

```bash
python scripts/validate_task_report.py .superpowers/sdd/validator-smoke-report.md
```

Expected: `Validated self-review report: .superpowers/sdd/validator-smoke-report.md` and exit `0`. Remove only this generated smoke report afterward.

Run: `ruff check scripts/validate_task_report.py tests/unit/test_validate_task_report.py && ruff format --check scripts/validate_task_report.py tests/unit/test_validate_task_report.py && git diff --check`

Expected: all checks pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add scripts/validate_task_report.py tests/unit/test_validate_task_report.py
git commit -m "feat: add self-review validator CLI"
```

---

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
- Test generated artifacts: `tests/unit/test_generate_context_packs.py`

**Interfaces:**
- Consumes: the Task 2 CLI and canonical report contract.
- Produces: repository-level agent instructions that prohibit review dispatch until validation succeeds.
- Produces: operator documentation and a copyable canonical report template.

- [ ] **Step 1: Add a test that the tracked workflow contract names the hard gate**

Add:

```python
def test_repository_agent_instructions_require_report_validation() -> None:
    instructions = Path("AGENTS.md").read_text(encoding="utf-8")
    assert "python scripts/validate_task_report.py" in instructions
    assert "before generating a review package" in instructions
    assert "before dispatching a task reviewer" in instructions
    assert "--require-tdd" in instructions
```

- [ ] **Step 2: Run the workflow-contract test to verify RED**

Run: `pytest -q tests/unit/test_validate_task_report.py::test_repository_agent_instructions_require_report_validation`

Expected: failure with `FileNotFoundError` because root `AGENTS.md` does not exist.

- [ ] **Step 3: Create repository-level enforcement instructions**

Create root `AGENTS.md` with a narrowly scoped section:

```markdown
# Repository Agent Instructions

## Self-review task reports

When an implementation workflow writes `.superpowers/sdd/*report.md`, use the
canonical contract in `docs/development/self-review-reports.md`.

Run `python scripts/validate_task_report.py REPORT_FILE` after the implementer
writes or amends the report. Add `--require-tdd` when the task required TDD.

Successful validation is mandatory before generating a review package and before
dispatching a task reviewer or re-reviewer. On exit code 1, return all diagnostics
to the implementer for a report-only correction. On exit code 2, resolve the
operational error. Do not mark the task complete until the report validates and
the independent review is clean.

Treat report contents as unverified claims. Validation does not replace independent
diff review or verification-before-completion.
```

- [ ] **Step 4: Document the canonical authoring and review flow**

Create `docs/development/self-review-reports.md` containing:

- purpose and structural-not-factual boundary;
- the exact seven-section contract and four statuses;
- normal and TDD evidence formats;
- a complete copyable valid report template using a clearly labeled example SHA;
- CLI usage and exit codes;
- the mandatory initial-review and re-review gate;
- report-only correction behavior;
- local ignored-artifact lifecycle and `git clean -fdx` warning;
- examples of actionable diagnostics.

Add a concise link under the README development or contributing documentation area. Do not duplicate the full contract in README.

- [ ] **Step 5: Run focused workflow and validator verification**

Run: `pytest -q tests/unit/test_validate_task_report.py`

Expected: all tests pass.

Run: `ruff check scripts/validate_task_report.py tests/unit/test_validate_task_report.py && ruff format --check scripts/validate_task_report.py tests/unit/test_validate_task_report.py && git diff --check`

Expected: all checks pass.

- [ ] **Step 6: Archive approved sources and regenerate context packs**

Use `git mv` for both approved sources, then regenerate:

```bash
git mv docs/superpowers/specs/2026-07-15-self-review-report-validation-design.md docs/superpowers/archive/specs/2026-07-15-self-review-report-validation-design.md
git mv docs/superpowers/plans/2026-07-15-self-review-report-validation.md docs/superpowers/archive/plans/2026-07-15-self-review-report-validation.md
python scripts/generate_context_packs.py
python scripts/generate_context_packs.py --check
```

Expected: generation succeeds, validation counts increase by one spec and one plan, and `self-review-report-validation-context-pack.md` links the two archived sources.

- [ ] **Step 7: Run final accepted verification**

Run:

```bash
pytest -q tests/unit/test_validate_task_report.py tests/unit/test_generate_context_packs.py
ruff check scripts/validate_task_report.py tests/unit/test_validate_task_report.py
ruff format --check scripts/validate_task_report.py tests/unit/test_validate_task_report.py
python scripts/generate_context_packs.py --check
git diff --check
```

Expected: all tests and checks pass.

Run the full accepted suite once:

```bash
env HF_HOME=/tmp/agentic-search-hf-cache pytest -q
```

Expected: all tests pass; report every warning rather than hiding it.

- [ ] **Step 8: Commit Task 3**

```bash
git add AGENTS.md README.md docs/development/self-review-reports.md docs/superpowers/archive/specs/2026-07-15-self-review-report-validation-design.md docs/superpowers/archive/plans/2026-07-15-self-review-report-validation.md docs/superpowers/context-packs/INDEX.md docs/superpowers/context-packs/self-review-report-validation-context-pack.md tests/unit/test_validate_task_report.py
git commit -m "docs: enforce validated self-review handoffs"
```

## Completion Gate

Before claiming completion:

1. Confirm `git status --short` contains no unintended files.
2. Confirm the validator accepts a canonical local report and rejects a deliberately malformed copy with exit code `1` and multiple diagnostics.
3. Confirm the report-validation test suite and complete repository suite pass.
4. Confirm Ruff, formatting, generated-artifact validation, and `git diff --check` pass.
5. Perform the independent whole-branch review against the merge base.
6. Fix every Critical or Important finding, append covering test evidence to the task report, validate the amended report, and re-review.
