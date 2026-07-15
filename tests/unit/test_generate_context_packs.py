from __future__ import annotations

from pathlib import Path

from scripts.generate_context_packs import (
    discover_sources,
    generate,
    normalize_source,
    pair_sources,
    validate_generated,
)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_discover_sources_combines_active_and_archive(tmp_path: Path) -> None:
    root = tmp_path / "superpowers"
    write(root / "specs/2026-07-01-active-design.md", "# Active\n")
    write(root / "archive/plans/2026-06-01-old.md", "# Old\n")

    generate(root, root / "context-packs")

    index = (root / "context-packs/INDEX.md").read_text(encoding="utf-8")
    assert index.count("../specs/2026-07-01-active-design.md") == 1
    assert index.count("../archive/plans/2026-06-01-old.md") == 1


def test_discover_sources_rejects_duplicate_filenames(tmp_path: Path) -> None:
    root = tmp_path / "superpowers"
    name = "2026-07-01-routing-design.md"
    active = root / "specs" / name
    archived = root / "archive/specs" / name
    write(active, "# Active Routing\n")
    write(archived, "# Archived Routing\n")

    try:
        discover_sources(root)
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("duplicate source filename was accepted")

    assert name in message
    assert str(active) in message
    assert str(archived) in message


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
        "routing-2026-07-01-spec-context-pack.md",
        "routing-2026-07-02-spec-context-pack.md",
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


def test_generate_strips_embedded_links_from_compacted_source(tmp_path: Path) -> None:
    root = tmp_path / "superpowers"
    write(
        root / "specs/2026-07-01-search-design.md",
        "# Search\n\n## Goal\n\nRead the [repository guide](../../../README.md).\n",
    )

    generate(root, root / "context-packs")

    pack = (root / "context-packs/search-context-pack.md").read_text(encoding="utf-8")
    assert "Read the repository guide." in pack
    assert "../../../README.md" not in pack
    assert validate_generated(root, root / "context-packs") == []


def test_validate_coverage_handles_same_basename_in_specs_and_plans(
    tmp_path: Path,
) -> None:
    root = tmp_path / "superpowers"
    name = "2026-07-01-routing.md"
    write(root / "specs" / name, "# Routing Spec\n")
    write(root / "plans" / name, "# Routing Plan\n")

    generate(root, root / "context-packs")

    index = (root / "context-packs/INDEX.md").read_text(encoding="utf-8")
    assert index.count(f"../specs/{name}") == 1
    assert index.count(f"../plans/{name}") == 1
    assert validate_generated(root, root / "context-packs") == []


def test_pair_sources_makes_same_date_collisions_unique(tmp_path: Path) -> None:
    paths = [
        ("spec", tmp_path / "specs/2026-07-01-routing-design.md"),
        ("spec", tmp_path / "other/2026-07-01-routing-design.md"),
        ("plan", tmp_path / "plans/2026-07-01-routing.md"),
    ]
    for kind, path in paths:
        write(path, f"# {kind.title()}\n")

    bundles = pair_sources(
        [normalize_source(path, kind) for kind, path in paths if kind == "spec"],
        [normalize_source(path, kind) for kind, path in paths if kind == "plan"],
    )

    names = [bundle.output_name for bundle in bundles]
    assert len(names) == len(set(names)) == 3
    assert all(name.startswith("routing-2026-07-01-") for name in names)


def test_render_pack_reserves_space_for_acceptance_after_many_tasks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "superpowers"
    tasks = "\n\n".join(
        f"## Task {number}\n\n- [ ] Step {number}\n\n```python\nprint({number})\n```"
        for number in range(1, 10)
    )
    write(
        root / "plans/2026-07-01-routing.md",
        f"# Routing\n\n{tasks}\n\n## Final Acceptance Checklist\n\n"
        + ("- [ ] All routing checks pass.\n" * 40),
    )

    generate(root, root / "context-packs")

    pack = (root / "context-packs/routing-context-pack.md").read_text(encoding="utf-8")
    assert "Final Acceptance Checklist" in pack
    assert "All routing checks pass." in pack
    assert "print(1)" not in pack
    assert "Section compacted" not in pack
    assert len(pack.splitlines()) < 100


def test_validate_generated_rejects_stale_owned_pack_and_placeholder(
    tmp_path: Path,
) -> None:
    root = tmp_path / "superpowers"
    output = root / "context-packs"
    write(root / "specs/2026-07-01-search-design.md", "# Search\n")
    generate(root, output)
    write(output / "stale-context-pack.md", "# Generated Context Pack\n")
    current = output / "search-context-pack.md"
    current.write_text(current.read_text() + "\nTODO: generated placeholder\n")

    errors = validate_generated(root, output)

    assert any("unexpected generated file" in error for error in errors)
    assert any("placeholder marker" in error for error in errors)


def test_markdown_parser_ignores_heading_syntax_inside_fences(tmp_path: Path) -> None:
    source_path = tmp_path / "2026-07-01-shell-design.md"
    write(
        source_path,
        "# Shell\n\n## Goal\n\nRun safely.\n\n```bash\n# not a heading\necho ok\n```\n",
    )

    source = normalize_source(source_path, "spec")

    assert [section.heading for section in source.sections] == ["Goal"]
    assert "# not a heading" in source.sections[0].body


def test_markdown_parser_requires_matching_fence_type_and_length(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "2026-07-01-shell-design.md"
    write(
        source_path,
        "# Shell\n\n## Goal\n\n````markdown\n```\n# still not a heading\n````\n",
    )

    source = normalize_source(source_path, "spec")

    assert [section.heading for section in source.sections] == ["Goal"]
    assert "# still not a heading" in source.sections[0].body


def test_category_matching_does_not_treat_substrings_as_words(tmp_path: Path) -> None:
    root = tmp_path / "superpowers"
    write(
        root / "plans/2026-07-01-results.md",
        "# Results\n\n## Contest Results\n\nA winner.\n\n"
        "## Final Verification\n\nRun the full suite.\n",
    )

    generate(root, root / "context-packs")

    pack = (root / "context-packs/results-context-pack.md").read_text(encoding="utf-8")
    assert "Final Verification" in pack
    assert "Run the full suite." in pack
