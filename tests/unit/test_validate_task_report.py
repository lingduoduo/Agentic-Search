from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from examples.validate_task_report import (
    GitInspectionError,
    main,
    parse_report,
    resolve_git_commit,
    validate_report,
)


def canonical_report(*, status: str = "DONE", commit: str = "abc1234") -> str:
    return f"""# Task 1 Report

## Status

{status}

## Commits

- `{commit}` implementation commit

## Implementation

Implemented the requested validator behavior.

## Files changed

- `examples/validate_task_report.py`

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


def messages(text: str, *, require_tdd: bool = False) -> list[str]:
    diagnostics = validate_report(
        parse_report(text),
        resolve_commit=lambda sha: sha in {"abc1234", "a" * 40},
        require_tdd=require_tdd,
    )
    return [diagnostic.message for diagnostic in diagnostics]


def tdd_report() -> str:
    evidence = """### RED

Command:

```text
pytest -q tests/unit/test_validate_task_report.py
```

Expected failure:

The validator module does not exist yet.

Result:

The focused suite failed with the expected import error.

### GREEN

Command:

```text
pytest -q tests/unit/test_validate_task_report.py
```

Result:

The focused suite passed with 20 tests.
"""
    start = canonical_report().index("### Test: focused unit suite")
    end = canonical_report().index("\n## Self-review")
    return canonical_report()[:start] + evidence + canonical_report()[end:]


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


@pytest.mark.parametrize(
    "status", ["DONE", "DONE_WITH_CONCERNS", "BLOCKED", "NEEDS_CONTEXT"]
)
def test_accepts_established_statuses(status: str) -> None:
    assert messages(canonical_report(status=status)) == []


@pytest.mark.parametrize("status", ["done", "COMPLETE", "", "DONE\nBLOCKED"])
def test_rejects_other_statuses(status: str) -> None:
    assert "invalid Status; expected exactly one workflow state" in messages(
        canonical_report(status=status)
    )


def test_duplicate_and_swapped_sections_are_reported() -> None:
    duplicate = canonical_report().replace(
        "## Implementation", "## Status\n\nDONE\n\n## Implementation"
    )
    assert "duplicate required section: Status" in messages(duplicate)

    swapped = (
        canonical_report()
        .replace("## Status", "## TEMP", 1)
        .replace("## Commits", "## Status", 1)
        .replace("## TEMP", "## Commits", 1)
    )
    assert "required sections are out of order" in messages(swapped)


def test_unknown_level_two_section_is_rejected() -> None:
    text = canonical_report().replace(
        "## Concerns", "## Notes\n\nAdditional notes.\n\n## Concerns"
    )
    assert "unexpected level-two section: Notes" in messages(text)


def test_empty_bodies_and_concerns_none_rules() -> None:
    empty = canonical_report().replace(
        "Implemented the requested validator behavior.\n\n## Files changed",
        "\n## Files changed",
    )
    assert "section has no substantive content" in messages(empty)
    assert messages(canonical_report()) == []


@pytest.mark.parametrize("section", ["Implementation", "Files changed", "Self-review"])
def test_none_is_not_substantive_outside_concerns(section: str) -> None:
    text = canonical_report()
    parsed = parse_report(text)
    target = next(item for item in parsed.sections if item.title == section)
    text = text.replace(target.body, "None", 1)
    assert "section has no substantive content" in messages(text)


def test_none_remains_valid_for_concerns() -> None:
    assert messages(canonical_report()) == []


@pytest.mark.parametrize("marker", ["TODO", "tbd", "FIXME", "placeholder"])
def test_rejects_standalone_placeholders(marker: str) -> None:
    text = canonical_report().replace(
        "Implemented the requested validator behavior.", marker
    )
    assert "section contains placeholder content" in messages(text)


@pytest.mark.parametrize("token", ["<COMMIT_SHA>", "[TEST RESULT]"])
def test_rejects_unexpanded_template_tokens(token: str) -> None:
    text = canonical_report().replace(
        "Implemented the requested validator behavior.", f"Implemented {token}."
    )
    assert "section contains unexpanded template token" in messages(text)


def test_ordinary_markdown_links_are_not_template_tokens() -> None:
    text = canonical_report().replace(
        "Implemented the requested validator behavior.",
        "Implemented behavior described in [guide](docs/guide.md).",
    )
    assert messages(text) == []


@pytest.mark.parametrize(
    "reference",
    [
        "See [API][api] for details.\n\n[api]: docs/api.md",
        "See [API][] for details.\n\n[API]: docs/api.md",
        "See [API] for details.\n\n[API]: docs/api.md",
    ],
)
def test_reference_style_markdown_links_are_not_template_tokens(reference: str) -> None:
    text = canonical_report().replace(
        "Implemented the requested validator behavior.", reference
    )
    assert messages(text) == []


def test_uppercase_template_token_without_reference_definition_is_rejected() -> None:
    text = canonical_report().replace(
        "Implemented the requested validator behavior.", "Implemented [FIELD]."
    )
    assert "section contains unexpanded template token" in messages(text)


@pytest.mark.parametrize(
    "line",
    ["- `HEAD` symbolic", "abc1234", "- abc1234 unquoted", "- `abc123` short"],
)
def test_rejects_invalid_commit_bullet_syntax(line: str) -> None:
    text = canonical_report().replace("- `abc1234` implementation commit", line)
    assert "invalid commit item; expected '- `SHA` optional subject'" in messages(text)


def test_accepts_full_sha_and_reports_unresolved_commit() -> None:
    assert messages(canonical_report(commit="a" * 40)) == []
    assert "commit does not resolve: deadbee" in messages(
        canonical_report(commit="deadbee")
    )


@pytest.mark.parametrize(
    ("old", "expected"),
    [
        ("Command:", "evidence block must contain exactly one Command: label"),
        ("Result:", "evidence block must contain exactly one Result: label"),
    ],
)
def test_missing_evidence_labels_are_reported(old: str, expected: str) -> None:
    text = canonical_report().replace(old, old.removesuffix(":"), 1)
    assert expected in messages(text)


def test_duplicate_evidence_labels_are_reported() -> None:
    text = canonical_report().replace("Result:", "Result:\n\nResult:", 1)
    assert "evidence block must contain exactly one Result: label" in messages(text)


def test_result_must_follow_command() -> None:
    text = canonical_report().replace(
        "Command:\n\n```text\npytest -q tests/unit/test_validate_task_report.py\n```\n\nResult:\n\nThe focused suite passed with 20 tests.",
        "Result:\n\nThe focused suite passed with 20 tests.\n\nCommand:\n\n```text\npytest -q tests/unit/test_validate_task_report.py\n```",
    )
    assert "evidence Command: must appear before Result:" in messages(text)


def test_empty_command_fence_and_empty_result_are_reported() -> None:
    empty_command = canonical_report().replace(
        "pytest -q tests/unit/test_validate_task_report.py", ""
    )
    assert "evidence Command: must be followed by a nonempty fenced block" in messages(
        empty_command
    )
    empty_result = canonical_report().replace(
        "The focused suite passed with 20 tests.\n\n## Self-review",
        "\n## Self-review",
    )
    assert "evidence Result: must contain nonempty prose" in messages(empty_result)


def test_rejects_unknown_evidence_block_title() -> None:
    text = canonical_report().replace(
        "### Test: focused unit suite", "### focused unit suite"
    )
    assert "invalid evidence block heading: focused unit suite" in messages(text)


def test_test_evidence_requires_at_least_one_block() -> None:
    text = canonical_report().replace(
        "### Test: focused unit suite", "Test execution summary"
    )
    assert "Test evidence must contain at least one evidence block" in messages(text)


def test_tdd_requires_red_green_and_expected_failure() -> None:
    errors = messages(canonical_report(), require_tdd=True)
    assert "missing TDD evidence block: RED" in errors
    assert "missing TDD evidence block: GREEN" in errors

    no_expected = tdd_report().replace("Expected failure:", "Expectation:")
    assert "RED must contain a nonempty Expected failure:" in messages(
        no_expected, require_tdd=True
    )

    empty_expected = tdd_report().replace(
        "The validator module does not exist yet.\n\nResult:", "Result:"
    )
    assert "RED must contain a nonempty Expected failure:" in messages(
        empty_expected, require_tdd=True
    )


def test_tdd_green_requires_passing_result() -> None:
    text = tdd_report().replace(
        "The focused suite passed with 20 tests.", "The focused suite was executed."
    )
    assert "GREEN Result: must state a passing outcome" in messages(
        text, require_tdd=True
    )


@pytest.mark.parametrize(
    "result",
    [
        "The suite did not pass.",
        "The tests were not all passing.",
        "The suite has not fully passed.",
        "The checks are not currently passing.",
        "The tests were not quite all currently passing.",
        "Exited with exit code 0, although assertions failed.",
        "20 passed. A later validation step reported an error.",
        "All checks are passing. One integration test is failing.",
        "0 passed, 1 failed.",
    ],
)
def test_tdd_green_rejects_negated_or_contradictory_passing_result(result: str) -> None:
    text = tdd_report().replace("The focused suite passed with 20 tests.", result)
    assert "GREEN Result: must state a passing outcome" in messages(
        text, require_tdd=True
    )


@pytest.mark.parametrize(
    "result", ["20 passed.", "All checks are passing.", "Exited with exit code 0."]
)
def test_tdd_green_accepts_unambiguous_passing_result(result: str) -> None:
    text = tdd_report().replace("The focused suite passed with 20 tests.", result)
    assert messages(text, require_tdd=True) == []


def test_tdd_green_allows_warnings_with_positive_pass_count() -> None:
    text = tdd_report().replace(
        "The focused suite passed with 20 tests.", "2644 passed, 6 warnings."
    )
    assert messages(text, require_tdd=True) == []


def test_result_requires_prose_not_only_a_fence() -> None:
    text = canonical_report().replace(
        "The focused suite passed with 20 tests.", "```text\npassed\n```"
    )
    assert "evidence Result: must contain nonempty prose" in messages(text)


def test_placeholders_inside_evidence_blocks_are_rejected() -> None:
    text = canonical_report().replace("The focused suite passed with 20 tests.", "TODO")
    assert "section contains placeholder content" in messages(text)


def test_valid_tdd_report_has_no_diagnostics() -> None:
    assert messages(tdd_report(), require_tdd=True) == []


def test_independent_violations_are_aggregated() -> None:
    text = canonical_report(status="COMPLETE", commit="deadbee").replace(
        "Implemented the requested validator behavior.", "TODO"
    )
    errors = messages(text)
    assert "invalid Status; expected exactly one workflow state" in errors
    assert "commit does not resolve: deadbee" in errors
    assert "section contains placeholder content" in errors


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
    git(repo, "config", "commit.gpgsign", "false")
    (repo / "tracked.txt").write_text("content\n", encoding="utf-8")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-q", "-m", "test commit")
    return repo, git(repo, "rev-parse", "HEAD")


def test_resolve_git_commit_accepts_full_and_unique_abbreviated_sha(
    tmp_path: Path,
) -> None:
    repo, sha = make_repo(tmp_path)

    assert resolve_git_commit(sha, cwd=repo) is True
    assert resolve_git_commit(sha[:7], cwd=repo) is True
    assert resolve_git_commit("deadbee", cwd=repo) is False


def test_resolve_git_commit_rejects_execution_outside_repository(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        GitInspectionError, match="unable to inspect commits outside a Git worktree"
    ):
        resolve_git_commit("deadbee", cwd=tmp_path)


def write_repo_report(repo: Path, sha: str, *, tdd: bool = False) -> Path:
    path = repo / "report.md"
    text = tdd_report() if tdd else canonical_report()
    path.write_text(text.replace("abc1234", sha[:7]), encoding="utf-8")
    return path


def test_cli_valid_report_prints_success_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, sha = make_repo(tmp_path)
    report = write_repo_report(repo, sha)

    assert main([str(report)]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"Validated self-review report: {report}\n"
    assert captured.err == ""


def test_cli_aggregates_contract_diagnostics_and_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, sha = make_repo(tmp_path)
    report = write_repo_report(repo, sha)
    report.write_text(
        report.read_text(encoding="utf-8")
        .replace("DONE", "COMPLETE", 1)
        .replace("Implemented the requested validator behavior.", "TODO", 1),
        encoding="utf-8",
    )

    assert main([str(report)]) == 1
    captured = capsys.readouterr()
    assert captured.out == (
        f"{report}:3: [Status] invalid Status; expected exactly one workflow state\n"
        f"{report}:11: [Implementation] section contains placeholder content\n"
    )
    assert captured.err == ""


@pytest.mark.parametrize("kind", ["missing", "invalid_utf8"])
def test_cli_input_errors_go_to_stderr_and_return_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str
) -> None:
    report = tmp_path / "report.md"
    if kind == "invalid_utf8":
        report.write_bytes(b"\xff")

    assert main([str(report)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"{report}: ERROR: ")


def test_cli_require_tdd_is_enforced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, sha = make_repo(tmp_path)
    report = write_repo_report(repo, sha)

    assert main(["--require-tdd", str(report)]) == 1
    captured = capsys.readouterr()
    assert "[Test evidence] missing TDD evidence block: RED" in captured.out
    assert "[Test evidence] missing TDD evidence block: GREEN" in captured.out

    write_repo_report(repo, sha, tdd=True)
    assert main(["--require-tdd", str(report)]) == 0


def test_cli_git_inspection_error_has_no_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "report.md"
    report.write_text(canonical_report(), encoding="utf-8")

    assert main([str(report)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        f"{report}: ERROR: unable to inspect commits outside a Git worktree\n"
    )


def test_cli_invalid_invocation_uses_argparse_exit_two() -> None:
    with pytest.raises(SystemExit) as caught:
        main([])
    assert caught.value.code == 2


def test_repository_agent_instructions_require_report_validation() -> None:
    instructions = Path("AGENTS.md").read_text(encoding="utf-8")
    assert "python examples/validate_task_report.py" in instructions
    assert "before generating a review package" in instructions
    assert "before dispatching a task reviewer" in instructions
    assert "--require-tdd" in instructions
