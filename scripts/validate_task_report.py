"""Parse and validate self-review task reports."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
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
MARKDOWN_LINK_RE = re.compile(r"\[[^\]\n]+\]\([^\)\n]*\)")
COMMAND_LABEL_RE = re.compile(r"(?m)^Command:[ \t]*$")
RESULT_LABEL_RE = re.compile(r"(?m)^Result:[ \t]*$")
EXPECTED_FAILURE_RE = re.compile(r"(?m)^Expected failure:[ \t]*(?:\n+)?(.+)$")
PASSING_RESULT_RE = re.compile(
    r"\b(?:pass|passed|passing|exit(?: code)?[ :=]+0)\b", re.IGNORECASE
)


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
    """Parse Markdown headings and bodies, ignoring headings inside fences."""
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
                (
                    len(heading_match.group(1)),
                    heading_match.group(2).strip(),
                    index,
                    index,
                )
            )

    sections: list[Section] = []
    for offset, (level, title, line, body_start) in enumerate(headings):
        body_end = (
            headings[offset + 1][2] - 1 if offset + 1 < len(headings) else len(lines)
        )
        body = "\n".join(lines[body_start:body_end]).strip()
        sections.append(Section(level=level, title=title, body=body, line=line))
    return ParsedReport(tuple(sections))


def validate_report(
    report: ParsedReport,
    *,
    resolve_commit: Callable[[str], bool],
    require_tdd: bool = False,
) -> tuple[Diagnostic, ...]:
    """Return every report-contract violation without performing I/O."""
    level_two = [section for section in report.sections if section.level == 2]
    by_title = {
        title: [section for section in level_two if section.title == title]
        for title in REQUIRED_SECTIONS
    }
    missing = [
        Diagnostic(0, title, f"missing required section: {title}")
        for title in REQUIRED_SECTIONS
        if not by_title[title]
    ]
    diagnostics: list[Diagnostic] = []

    for title in REQUIRED_SECTIONS:
        matches = by_title[title]
        for duplicate in matches[1:]:
            diagnostics.append(
                Diagnostic(
                    duplicate.line, title, f"duplicate required section: {title}"
                )
            )

    first_sections = [
        by_title[title][0] for title in REQUIRED_SECTIONS if by_title[title]
    ]
    if len(first_sections) > 1 and any(
        left.line > right.line
        for left, right in zip(first_sections, first_sections[1:])
    ):
        diagnostics.append(
            Diagnostic(
                min(section.line for section in first_sections),
                None,
                "required sections are out of order",
            )
        )

    evidence_blocks = _evidence_blocks(report, by_title["Test evidence"])
    for title in REQUIRED_SECTIONS:
        for section in by_title[title]:
            substantive = bool(section.body.strip())
            if title == "Test evidence":
                substantive = substantive or bool(evidence_blocks)
            if not substantive and not (
                title == "Concerns" and section.body.strip() == "None"
            ):
                diagnostics.append(
                    Diagnostic(
                        section.line, title, "section has no substantive content"
                    )
                )
            diagnostics.extend(_placeholder_diagnostics(section))

    if by_title["Status"]:
        status = by_title["Status"][0]
        if status.body.strip() not in VALID_STATUSES:
            diagnostics.append(
                Diagnostic(
                    status.line,
                    status.title,
                    "invalid Status; expected exactly one workflow state",
                )
            )

    if by_title["Commits"]:
        diagnostics.extend(_validate_commits(by_title["Commits"][0], resolve_commit))

    if by_title["Test evidence"]:
        diagnostics.extend(_validate_evidence_blocks(evidence_blocks, require_tdd))

    return tuple(
        missing
        + sorted(
            diagnostics,
            key=lambda item: (item.line, item.section or "", item.message),
        )
    )


def _placeholder_diagnostics(section: Section) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    if any(PLACEHOLDER_LINE_RE.fullmatch(line) for line in section.body.splitlines()):
        diagnostics.append(
            Diagnostic(
                section.line, section.title, "section contains placeholder content"
            )
        )
    without_links = MARKDOWN_LINK_RE.sub("", section.body)
    if TEMPLATE_TOKEN_RE.search(without_links):
        diagnostics.append(
            Diagnostic(
                section.line,
                section.title,
                "section contains unexpanded template token",
            )
        )
    return diagnostics


def _validate_commits(
    section: Section, resolve_commit: Callable[[str], bool]
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    valid_shas: list[str] = []
    for offset, line in enumerate(section.body.splitlines(), start=1):
        if not line.strip():
            continue
        match = COMMIT_ITEM_RE.fullmatch(line)
        if match is None:
            diagnostics.append(
                Diagnostic(
                    section.line + offset,
                    section.title,
                    "invalid commit item; expected '- `SHA` optional subject'",
                )
            )
            continue
        sha = match.group(1)
        valid_shas.append(sha)
        if not resolve_commit(sha):
            diagnostics.append(
                Diagnostic(
                    section.line + offset,
                    section.title,
                    f"commit does not resolve: {sha}",
                )
            )
    if not valid_shas:
        diagnostics.append(
            Diagnostic(
                section.line, section.title, "Commits must list at least one SHA"
            )
        )
    return diagnostics


def _evidence_blocks(
    report: ParsedReport, test_evidence_sections: list[Section]
) -> list[Section]:
    if not test_evidence_sections:
        return []
    parent = test_evidence_sections[0]
    blocks: list[Section] = []
    for section in report.sections:
        if section.line <= parent.line:
            continue
        if section.level <= 2:
            break
        if section.level == 3:
            blocks.append(section)
    return blocks


def _validate_evidence_blocks(
    blocks: list[Section], require_tdd: bool
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    if not blocks:
        diagnostics.append(
            Diagnostic(
                0,
                "Test evidence",
                "Test evidence must contain at least one evidence block",
            )
        )
    for block in blocks:
        diagnostics.extend(_placeholder_diagnostics(block))
        if block.title not in {"RED", "GREEN"} and not block.title.startswith("Test: "):
            diagnostics.append(
                Diagnostic(
                    block.line,
                    "Test evidence",
                    f"invalid evidence block heading: {block.title}",
                )
            )
        command_labels = list(COMMAND_LABEL_RE.finditer(block.body))
        result_labels = list(RESULT_LABEL_RE.finditer(block.body))
        if len(command_labels) != 1:
            diagnostics.append(
                Diagnostic(
                    block.line,
                    block.title,
                    "evidence block must contain exactly one Command: label",
                )
            )
        elif not _has_nonempty_command_fence(block.body, command_labels[0].end()):
            diagnostics.append(
                Diagnostic(
                    block.line,
                    block.title,
                    "evidence Command: must be followed by a nonempty fenced block",
                )
            )
        if len(result_labels) != 1:
            diagnostics.append(
                Diagnostic(
                    block.line,
                    block.title,
                    "evidence block must contain exactly one Result: label",
                )
            )
        elif not _result_text(block.body, result_labels[0].end()):
            diagnostics.append(
                Diagnostic(
                    block.line,
                    block.title,
                    "evidence Result: must contain nonempty prose",
                )
            )

    if require_tdd:
        for title in ("RED", "GREEN"):
            if not any(block.title == title for block in blocks):
                diagnostics.append(
                    Diagnostic(
                        0,
                        "Test evidence",
                        f"missing TDD evidence block: {title}",
                    )
                )
        red = next((block for block in blocks if block.title == "RED"), None)
        if red is not None:
            match = EXPECTED_FAILURE_RE.search(red.body)
            explanation = ""
            if match is not None:
                start = match.start(1)
                result_label = RESULT_LABEL_RE.search(red.body, start)
                end = (
                    result_label.start() if result_label is not None else len(red.body)
                )
                explanation = _prose_text(red.body[start:end])
            if not explanation:
                diagnostics.append(
                    Diagnostic(
                        red.line,
                        red.title,
                        "RED must contain a nonempty Expected failure:",
                    )
                )
        green = next((block for block in blocks if block.title == "GREEN"), None)
        if green is not None:
            matches = list(RESULT_LABEL_RE.finditer(green.body))
            result = (
                _result_text(green.body, matches[0].end()) if len(matches) == 1 else ""
            )
            if not PASSING_RESULT_RE.search(result):
                diagnostics.append(
                    Diagnostic(
                        green.line,
                        green.title,
                        "GREEN Result: must state a passing outcome",
                    )
                )
    return diagnostics


def _has_nonempty_command_fence(body: str, label_end: int) -> bool:
    remainder = body[label_end:].lstrip(" \t\r\n")
    lines = remainder.splitlines()
    if not lines:
        return False
    opener = FENCE_RE.match(lines[0])
    if opener is None:
        return False
    marker = opener.group(1)
    contents: list[str] = []
    for line in lines[1:]:
        closing = FENCE_RE.match(line)
        if (
            closing
            and closing.group(1)[0] == marker[0]
            and len(closing.group(1)) >= len(marker)
        ):
            return bool("\n".join(contents).strip())
        contents.append(line)
    return False


def _result_text(body: str, label_end: int) -> str:
    return _prose_text(body[label_end:])


def _prose_text(text: str) -> str:
    prose: list[str] = []
    fence: tuple[str, int] | None = None
    for line in text.splitlines():
        match = FENCE_RE.match(line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = (marker[0], len(marker))
            elif marker[0] == fence[0] and len(marker) >= fence[1]:
                fence = None
            continue
        if fence is None and line.strip():
            prose.append(line.strip())
    return "\n".join(prose)
