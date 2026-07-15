from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


DATE_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)$")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
PREFERRED_HEADINGS = (
    "goal",
    "overview",
    "purpose",
    "outcome",
    "decision",
    "architecture",
    "component",
    "constraint",
    "requirement",
    "scope",
    "task",
    "implementation",
    "verification",
    "acceptance",
    "test",
    "risk",
    "open question",
    "non-goal",
)
SECTION_LIMIT = 1_600
MAX_SECTIONS = 8
GENERATED_MARKER = "# Generated Context Pack"


@dataclass(frozen=True)
class Section:
    heading: str
    body: str


@dataclass(frozen=True)
class SourceDoc:
    path: Path
    kind: Literal["spec", "plan"]
    date: str
    topic: str
    title: str
    introduction: str
    sections: tuple[Section, ...]


@dataclass(frozen=True)
class TopicBundle:
    topic: str
    specs: tuple[SourceDoc, ...]
    plans: tuple[SourceDoc, ...]
    output_name: str

    @property
    def sources(self) -> tuple[SourceDoc, ...]:
        return tuple(
            sorted((*self.specs, *self.plans), key=lambda item: item.path.name)
        )

    @property
    def date(self) -> str:
        return min(source.date for source in self.sources)


def _parse_markdown(text: str) -> tuple[str, str, tuple[Section, ...]]:
    title = "Untitled"
    introduction: list[str] = []
    sections: list[Section] = []
    current_heading: str | None = None
    current_body: list[str] = []

    for line in text.splitlines():
        match = HEADING.match(line)
        if match:
            if current_heading is not None:
                sections.append(
                    Section(current_heading, "\n".join(current_body).strip())
                )
            elif current_body:
                introduction.extend(current_body)
            if len(match.group(1)) == 1 and title == "Untitled":
                title = match.group(2).strip()
                current_heading = None
                current_body = []
            else:
                current_heading = match.group(2).strip()
                current_body = []
            continue
        current_body.append(line)

    if current_heading is not None:
        sections.append(Section(current_heading, "\n".join(current_body).strip()))
    elif current_body:
        introduction.extend(current_body)

    return title, "\n".join(introduction).strip(), tuple(sections)


def normalize_source(path: Path, kind: str) -> SourceDoc:
    if kind not in {"spec", "plan"}:
        raise ValueError(f"unsupported source kind: {kind}")
    match = DATE_PREFIX.match(path.stem)
    if not match:
        raise ValueError(f"source filename lacks YYYY-MM-DD prefix: {path.name}")
    date, topic = match.groups()
    if kind == "spec" and topic.endswith("-design"):
        topic = topic[: -len("-design")]
    title, introduction, sections = _parse_markdown(path.read_text(encoding="utf-8"))
    return SourceDoc(
        path=path,
        kind=kind,  # type: ignore[arg-type]
        date=date,
        topic=topic,
        title=title,
        introduction=introduction,
        sections=sections,
    )


def discover_sources(source_root: Path) -> tuple[list[SourceDoc], list[SourceDoc]]:
    specs = [
        normalize_source(path, "spec")
        for path in sorted((source_root / "specs").glob("*.md"))
    ]
    plans = [
        normalize_source(path, "plan")
        for path in sorted((source_root / "plans").glob("*.md"))
    ]
    return specs, plans


def pair_sources(specs: list[SourceDoc], plans: list[SourceDoc]) -> list[TopicBundle]:
    by_topic: dict[str, dict[str, list[SourceDoc]]] = {}
    for source in (*specs, *plans):
        by_topic.setdefault(source.topic, {"spec": [], "plan": []})[source.kind].append(
            source
        )

    bundles: list[TopicBundle] = []
    for topic, grouped in by_topic.items():
        topic_specs = sorted(grouped["spec"], key=lambda source: source.path.name)
        topic_plans = sorted(grouped["plan"], key=lambda source: source.path.name)
        if len(topic_specs) <= 1 and len(topic_plans) <= 1:
            bundles.append(
                TopicBundle(
                    topic,
                    tuple(topic_specs),
                    tuple(topic_plans),
                    f"{topic}-context-pack.md",
                )
            )
            continue

        for source in (*topic_specs, *topic_plans):
            bundles.append(
                TopicBundle(
                    topic,
                    (source,) if source.kind == "spec" else (),
                    (source,) if source.kind == "plan" else (),
                    f"{topic}-{source.date}-context-pack.md",
                )
            )

    return sorted(bundles, key=lambda bundle: (bundle.topic, bundle.output_name))


def _compact(text: str, limit: int = SECTION_LIMIT) -> str:
    text = re.sub(r"!?\[([^]]+)]\([^)]+\)", r"\1", text).strip()
    if len(text) <= limit:
        return text
    boundary = text.rfind("\n", 0, limit)
    if boundary < limit // 2:
        boundary = text.rfind(" ", 0, limit)
    return text[: max(boundary, 0)].rstrip() + "\n\n_[Section compacted.]_"


def _selected_sections(source: SourceDoc) -> tuple[Section, ...]:
    selected = tuple(
        section
        for section in source.sections
        if section.body
        and any(token in section.heading.lower() for token in PREFERRED_HEADINGS)
    )
    if selected:
        return selected[:MAX_SECTIONS]
    if source.introduction:
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", source.introduction)
            if paragraph.strip()
        ]
        return (Section("Overview", "\n\n".join(paragraphs[:2])),)
    first_nonempty = next(
        (section for section in source.sections if section.body), None
    )
    return (first_nonempty,) if first_nonempty else ()


def _source_link(source: SourceDoc) -> str:
    return f"../{source.kind}s/{source.path.name}"


def render_pack(bundle: TopicBundle) -> str:
    title = (
        bundle.sources[0].title
        if len(bundle.sources) == 1
        else bundle.topic.replace("-", " ").title()
    )
    lines = [GENERATED_MARKER, "", f"# {title}", "", "## Sources", ""]
    for source in bundle.sources:
        label = "Specification" if source.kind == "spec" else "Plan"
        lines.append(f"- [{label}: {source.path.name}]({_source_link(source)})")

    for source in bundle.sources:
        label = "Specification" if source.kind == "spec" else "Implementation Plan"
        lines.extend(["", f"## {label} Context", ""])
        selected = _selected_sections(source)
        if not selected:
            lines.append(
                "_No summary content was available beyond the document title._"
            )
            continue
        for section in selected:
            lines.extend(
                [
                    f"### {section.heading}",
                    "",
                    _compact(section.body),
                    "",
                ]
            )
        while lines and lines[-1] == "":
            lines.pop()

    lines.extend(
        [
            "",
            "## Context Boundary",
            "",
            "This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.",
            "",
        ]
    )
    return "\n".join(lines)


def render_index(bundles: list[TopicBundle]) -> str:
    lines = [
        "# Context Pack Index",
        "",
        "Focused context packs generated from `docs/superpowers/specs` and `docs/superpowers/plans`.",
    ]
    by_date: dict[str, list[TopicBundle]] = {}
    for bundle in bundles:
        by_date.setdefault(bundle.date, []).append(bundle)

    for date in sorted(by_date):
        lines.extend(["", f"## {date}", ""])
        for bundle in sorted(by_date[date], key=lambda item: item.output_name):
            lines.append(
                f"### [{bundle.topic.replace('-', ' ').title()}]({bundle.output_name})"
            )
            lines.append("")
            for source in bundle.sources:
                label = "Spec" if source.kind == "spec" else "Plan"
                lines.append(f"- [{label}: {source.title}]({_source_link(source)})")
            lines.append("")
        while lines and lines[-1] == "":
            lines.pop()
    lines.append("")
    return "\n".join(lines)


def _expected_outputs(source_root: Path) -> dict[str, str]:
    specs, plans = discover_sources(source_root)
    bundles = pair_sources(specs, plans)
    rendered = {bundle.output_name: render_pack(bundle) for bundle in bundles}
    rendered["INDEX.md"] = render_index(bundles)
    return rendered


def generate(source_root: Path, output_dir: Path) -> list[Path]:
    rendered = _expected_outputs(source_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_names = set(rendered)
    for stale in output_dir.glob("*-context-pack.md"):
        if stale.name not in expected_names and stale.read_text(
            encoding="utf-8"
        ).startswith(GENERATED_MARKER):
            stale.unlink()

    written: list[Path] = []
    for name in sorted(rendered, key=lambda item: (item == "INDEX.md", item)):
        path = output_dir / name
        path.write_text(rendered[name], encoding="utf-8")
        written.append(path)
    return written


def validate_generated(source_root: Path, output_dir: Path) -> list[str]:
    errors: list[str] = []
    expected = _expected_outputs(source_root)
    for name, content in expected.items():
        path = output_dir / name
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            errors.append(f"generated file is out of date: {path}")

    for path in sorted(output_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            errors.append(f"generated file is empty: {path}")
        for target in MARKDOWN_LINK.findall(text):
            target = target.split("#", 1)[0]
            if not target or "://" in target:
                continue
            if not (path.parent / target).resolve().exists():
                errors.append(f"broken link in {path}: {target}")

    index_path = output_dir / "INDEX.md"
    if index_path.exists():
        index = index_path.read_text(encoding="utf-8")
        specs, plans = discover_sources(source_root)
        for source in (*specs, *plans):
            link = _source_link(source)
            count = index.count(link)
            if count != 1:
                errors.append(f"source appears {count} times in INDEX.md: {link}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compact spec/plan context packs"
    )
    parser.add_argument("--source-root", type=Path, default=Path("docs/superpowers"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/superpowers/context-packs")
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check:
        errors = validate_generated(args.source_root, args.output_dir)
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        specs, plans = discover_sources(args.source_root)
        print(f"Validated {len(specs)} specs and {len(plans)} plans")
        return 0
    generated = generate(args.source_root, args.output_dir)
    print(f"Generated {len(generated) - 1} context packs and INDEX.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
