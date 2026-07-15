from __future__ import annotations

import pytest

from scripts.validate_task_report import parse_report, validate_report


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
