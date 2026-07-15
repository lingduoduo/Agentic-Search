# Context Packing Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a hierarchical index and a compact context pack for every specification and implementation plan under `docs/superpowers/`.

**Architecture:** A dependency-free Python generator discovers source Markdown files, normalizes date-prefixed filenames into topic slugs, pairs specs with plans, extracts concise source-backed sections, and writes deterministic Markdown artifacts. Unit tests exercise discovery, pairing, collision handling, pack rendering, and index completeness before the generator runs against the repository corpus.

**Tech Stack:** Python 3 standard library, pytest, Markdown.

## Global Constraints

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

- [ ] **Step 2: Run tests and confirm the missing-module failure**

Run: `pytest tests/unit/test_generate_context_packs.py -q`

Expected: test collection fails because `scripts.generate_context_packs` does not exist.

- [ ] **Step 3: Implement source discovery and pairing**

Use frozen dataclasses for source metadata and topic bundles. Strip the leading `YYYY-MM-DD-` prefix and a spec's trailing `-design`; pair on the remaining slug. For collisions, retain every source and add a date suffix to ambiguous output filenames.

```python
@dataclass(frozen=True)
class SourceDoc:
    path: Path
    kind: Literal["spec", "plan"]
    date: str
    topic: str
    title: str
    sections: tuple[Section, ...]


@dataclass(frozen=True)
class TopicBundle:
    topic: str
    specs: tuple[SourceDoc, ...]
    plans: tuple[SourceDoc, ...]
    output_name: str
```

- [ ] **Step 4: Implement compact Markdown rendering**

Render source links followed by bounded summaries selected from headings matching goals, overview, decisions, architecture, constraints, tasks, verification, acceptance criteria, risks, and open questions. When no preferred heading exists, include the first non-empty introductory paragraphs. Preserve command blocks only under verification-related headings and cap each extracted section to keep packs focused.

- [ ] **Step 5: Implement generation and CLI entry point**

Expose `--source-root` and `--output-dir`, defaulting to `docs/superpowers` and `docs/superpowers/context-packs`. Write packs in sorted order, write `INDEX.md` last, and remove only stale `*-context-pack.md` files previously owned by this generator.

```python
if __name__ == "__main__":
    args = parse_args()
    generated = generate(args.source_root, args.output_dir)
    print(f"Generated {len(generated) - 1} context packs and INDEX.md")
```

- [ ] **Step 6: Run focused tests**

Run: `pytest tests/unit/test_generate_context_packs.py -q`

Expected: all context-pack generator tests pass.

- [ ] **Step 7: Commit the generator**

```bash
git add scripts/generate_context_packs.py tests/unit/test_generate_context_packs.py
git commit -m "feat: add deterministic context pack generator"
```

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

Expected: tests pass; check mode reports every source indexed exactly once, every relative link resolves, every pack is non-empty, and generated files match a fresh render.

- [ ] **Step 5: Inspect representative artifacts**

Inspect `INDEX.md`, one matched pack, one spec-only pack, and one plan-only pack. Confirm headings are readable, summaries remain source-backed, and links resolve from the generated directory.

- [ ] **Step 6: Run repository hygiene checks**

Run: `git diff --check`

Expected: exits 0 with no whitespace errors.

- [ ] **Step 7: Commit generated artifacts**

```bash
git add docs/superpowers/context-packs
git commit -m "docs: generate context packs for specs and plans"
```
