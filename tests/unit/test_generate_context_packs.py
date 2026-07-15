from __future__ import annotations

from pathlib import Path

from scripts.generate_context_packs import (
    generate,
    normalize_source,
    pair_sources,
    validate_generated,
)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_generate_pairs_sources_and_indexes_each_once(tmp_path: Path) -> None:
    root = tmp_path / "superpowers"
    write(
        root / "specs/2026-07-01-search-design.md",
        "# Search\n\n## Goal\n\nFast search.\n",
    )
    write(
        root / "plans/2026-07-02-search.md",
        "# Search Plan\n\n## Tasks\n\n1. Build it.\n",
    )

    written = generate(root, root / "context-packs")

    index = (root / "context-packs/INDEX.md").read_text(encoding="utf-8")
    pack = (root / "context-packs/search-context-pack.md").read_text(encoding="utf-8")
    assert index.count("2026-07-01-search-design.md") == 1
    assert index.count("2026-07-02-search.md") == 1
    assert "Fast search." in pack
    assert "Build it." in pack
    assert len(written) == 2


def test_normalize_source_uses_date_independent_topic(tmp_path: Path) -> None:
    spec_path = tmp_path / "2026-07-01-agent-search-design.md"
    plan_path = tmp_path / "2026-07-09-agent-search.md"
    write(spec_path, "# Agent Search\n")
    write(plan_path, "# Agent Search Plan\n")

    spec = normalize_source(spec_path, "spec")
    plan = normalize_source(plan_path, "plan")

    assert spec.date == "2026-07-01"
    assert plan.date == "2026-07-09"
    assert spec.topic == plan.topic == "agent-search"


def test_pair_sources_keeps_unmatched_and_disambiguates_collisions(
    tmp_path: Path,
) -> None:
    spec_one = tmp_path / "2026-07-01-routing-design.md"
    spec_two = tmp_path / "2026-07-02-routing-design.md"
    plan = tmp_path / "2026-07-03-planning.md"
    write(spec_one, "# Routing One\n")
    write(spec_two, "# Routing Two\n")
    write(plan, "# Planning\n")

    bundles = pair_sources(
        [normalize_source(spec_one, "spec"), normalize_source(spec_two, "spec")],
        [normalize_source(plan, "plan")],
    )

    assert [bundle.output_name for bundle in bundles] == [
        "planning-context-pack.md",
        "routing-2026-07-01-context-pack.md",
        "routing-2026-07-02-context-pack.md",
    ]
    assert bundles[0].specs == ()
    assert bundles[0].plans[0].topic == "planning"


def test_generate_uses_intro_when_preferred_headings_are_absent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "superpowers"
    write(
        root / "specs/2026-07-01-plain-design.md",
        "# Plain\n\nThis introduction explains the feature.\n\n"
        "A second paragraph adds context.\n\n## Appendix\n\nIgnore details.\n",
    )

    generate(root, root / "context-packs")

    pack = (root / "context-packs/plain-context-pack.md").read_text(encoding="utf-8")
    assert "This introduction explains the feature." in pack
    assert "A second paragraph adds context." in pack
    assert "Ignore details." not in pack


def test_generate_is_deterministic_and_removes_stale_owned_packs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "superpowers"
    output = root / "context-packs"
    write(root / "plans/2026-07-02-zeta.md", "# Zeta\n")
    write(root / "plans/2026-07-01-alpha.md", "# Alpha\n")
    write(output / "stale-context-pack.md", "# Generated Context Pack\n")
    write(output / "notes.md", "keep me\n")

    generate(root, output)
    first = {path.name: path.read_text() for path in output.iterdir()}
    generate(root, output)
    second = {path.name: path.read_text() for path in output.iterdir()}

    assert first == second
    assert "stale-context-pack.md" not in second
    assert second["notes.md"] == "keep me\n"
    assert second["INDEX.md"].index("alpha-context-pack.md") < second["INDEX.md"].index(
        "zeta-context-pack.md"
    )


def test_validate_generated_detects_drift_and_broken_links(tmp_path: Path) -> None:
    root = tmp_path / "superpowers"
    output = root / "context-packs"
    write(root / "specs/2026-07-01-search-design.md", "# Search\n")
    generate(root, output)

    assert validate_generated(root, output) == []

    (output / "search-context-pack.md").write_text(
        "# Generated Context Pack\n\n[missing](../specs/missing.md)\n",
        encoding="utf-8",
    )
    errors = validate_generated(root, output)
    assert any("out of date" in error for error in errors)
    assert any("broken link" in error for error in errors)
