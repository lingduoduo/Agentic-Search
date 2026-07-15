# Generated Context Pack

# Context Packing Artifacts

## Sources

- [Specification: 2026-07-15-context-packing-artifacts-design.md](../specs/2026-07-15-context-packing-artifacts-design.md)
- [Plan: 2026-07-15-context-packing-artifacts.md](../plans/2026-07-15-context-packing-artifacts.md)

## Specification Context

### Goal

Create focused, navigable context packs for every design specification and implementation plan under `docs/superpowers/`. The packs should let an agent load the context for one topic without flooding its context window with the complete documentation corpus.

### Scope

- Index every Markdown file directly under `docs/superpowers/specs/` and `docs/superpowers/plans/`.
- Match specifications and plans by their date-independent topic slug, removing the specification's trailing `-design` suffix.
- Create one compact context pack for each matched topic.
- Create a standalone pack when a specification or plan has no counterpart.
- Preserve all source specifications and plans unchanged.

### Non-Goals

- Changing or correcting existing specifications and plans.
- Determining whether planned work has been implemented.
- Creating implementation plans for the context-packing machinery itself beyond what is needed to generate the requested artifacts.
- Loading every pack simultaneously into an agent session.

## Implementation Plan Context

### Global Constraints

- Preserve all source specifications and plans unchanged.
- Summaries must contain only information supported by source documents.
- Do not infer implementation status from the current codebase.
- Generated links must be relative Markdown links.
- Source discovery and output ordering must be deterministic.

---

### Task 1: Deterministic Context-Pack Generator

**Files:**
- Create: `scripts/generate_context_packs.py`
- Create: `tests/unit/test_generate_context_packs.py`

**Interfaces:**
- Consumes: Markdown files in `<root>/specs/*.md` and `<root>/plans/*.md`.
- Produces: `normalize_source(path: Path, kind: str) -> SourceDoc`, `pair_sources(specs: list[SourceDoc], plans: list[SourceDoc]) -> list[TopicBundle]`, `render_pack(bundle: TopicBundle) -> str`, `render_index(bundles: list[TopicBundle]) -> str`, and `generate(source_root: Path, output_dir: Path) -> list[Path]`.

- [ ] **Step 1: Write failing unit tests**

Create fixtures in pytest's `tmp_path` covering a matched spec/plan pair, unmatched sources, different dates with the same topic, duplicate normalized slugs, heading-based summary extraction, relative links, deterministic ordering, and exact source coverage in the index.

```python
def test_generate_pairs_sources_and_indexes_each_once(tmp_path: Path) -> None:
    root = tmp_path / "superpowers"
    write(root / "specs/2026-07-01-search-design.md", "# Search\n\n## Goal\n\nFast search.\n")
    write(root / "plans/2026-07-02-search.md", "# Search Plan\n\n## Tasks\n\n1. Build it.\n")

    written = generate(root, root / "context-packs")

    index = (root / "context-packs/INDEX.md").read_text()
    pack = (root / "context-packs/search-context-pack.md").read_text()
    assert index.count("2026-07-01-search-design.md") == 1
    assert index.count("2026-07-02-search.md") == 1
    assert "Fast search." in pack
    assert "Build it." in pack
    assert len(written) == 2
```

_[Section compacted.]_

### Task 2: Generate and Validate Repository Context Packs

**Files:**
- Create: `docs/superpowers/context-packs/INDEX.md`
- Create: `docs/superpowers/context-packs/*-context-pack.md`

**Interfaces:**
- Consumes: `generate(source_root: Path, output_dir: Path) -> list[Path]` from Task 1 and every source file under `docs/superpowers/specs/` and `docs/superpowers/plans/`.
- Produces: a complete navigable context-pack corpus whose source coverage can be checked mechanically.

- [ ] **Step 1: Record source checksums**

Run:

```bash
find docs/superpowers/specs docs/superpowers/plans -type f -name '*.md' -print0 | sort -z | xargs -0 shasum -a 256 > /tmp/context-pack-source-checksums.before
```

Expected: one checksum line per source document.

- [ ] **Step 2: Generate all packs**

Run: `python scripts/generate_context_packs.py`

Expected: output reports the generated pack count and `INDEX.md` exists.

- [ ] **Step 3: Verify source files are unchanged**

Run:

```bash
find docs/superpowers/specs docs/superpowers/plans -type f -name '*.md' -print0 | sort -z | xargs -0 shasum -a 256 > /tmp/context-pack-source-checksums.after
diff -u /tmp/context-pack-source-checksums.before /tmp/context-pack-source-checksums.after
```

Expected: `diff` exits 0 with no output.

- [ ] **Step 4: Validate generated coverage and links**

Run: `pytest tests/unit/test_generate_context_packs.py -q`

Then run the generator's repository validation mode:

```bash
python scripts/generate_context_packs.py --check
```

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
